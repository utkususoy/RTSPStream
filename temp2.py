import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

# --- 1. Konfigürasyon Sınıfı ---
@dataclass
class RacetrackConfig:
    """Racetrack tespit algoritması için konfigüre edilebilir parametreler."""
    max_length_nm: float = 120.0      # Maksimum uzun bacak mesafesi (NM)
    max_width_nm: float = 60.0        # Maksimum kısa bacak/dönüş çapı (NM)
    max_time_min: float = 90.0        # Maksimum tur süresi (60x120 için süre artırıldı)
    turn_rate_threshold: float = 0.5  # Saniyede derece cinsinden dönüş eşiği
    heading_tolerance: float = 15.0   # Zıt yönlülük kuralı için tolerans (Derece)
    closure_tolerance_nm: float = 15.0 # Tur tamamlama için başlangıç/bitiş yakınlık toleransı

# --- 2. Yardımcı Matematiksel Fonksiyonlar ---
def get_heading_diff(h1: float, h2: float) -> float:
    """İki rota arasındaki en kısa açı farkını hesaplar (Döngüsel Matematik)."""
    diff = (h1 - h2 + 180) % 360 - 180
    return diff

def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki GPS koordinatı arasındaki mesafeyi Deniz Mili (NM) cinsinden hesaplar."""
    R = 3440.065 # Dünyanın yarıçapı (Deniz Mili)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) * math.sin(d_lon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- 3. Ana Tespit Sınıfı ---
class CapRacetrackDetector:
    def __init__(self, config: RacetrackConfig):
        self.config = config

    def _segment_flight(self, telemetry: List[Dict]) -> List[Dict]:
        """Telemetriyi 'Straight' (S) ve 'Turn' (T) bacaklarına ayırır."""
        segments = []
        current_segment = {"type": None, "points": [], "start_time": None, "end_time": None}
        
        for i in range(1, len(telemetry)):
            prev_pt = telemetry[i-1]
            curr_pt = telemetry[i]
            
            # Zaman ve rota farkı (Turn Rate)
            dt = curr_pt['timestamp'] - prev_pt['timestamp']
            if dt == 0: dt = 1
            
            heading_diff = get_heading_diff(curr_pt['heading'], prev_pt['heading'])
            turn_rate = abs(heading_diff) / dt
            
            # Segment tipini belirle
            point_type = 'T' if turn_rate > self.config.turn_rate_threshold else 'S'
            
            if current_segment["type"] is None:
                current_segment["type"] = point_type
                current_segment["start_time"] = prev_pt['timestamp']
            
            # Tip değişirse yeni segmente geç (Filtreleme burada basitleştirilmiştir)
            if point_type != current_segment["type"]:
                current_segment["end_time"] = prev_pt['timestamp']
                segments.append(current_segment)
                current_segment = {"type": point_type, "points": [], "start_time": prev_pt['timestamp']}
            
            current_segment["points"].append(curr_pt)
            
        if current_segment["points"]:
            current_segment["end_time"] = telemetry[-1]['timestamp']
            segments.append(current_segment)
            
        return segments

    def detect(self, telemetry: List[Dict]) -> bool:
        """Telemetri verisinde kurallara uyan bir S-T-S-T patemi arar."""
        segments = self._segment_flight(telemetry)
        
        # En az S-T-S-T dizilimi için 4 segment olmalı
        if len(segments) < 4:
            return False
            
        # Basit bir pattern matching: S-T-S-T ardışıklığını ara
        for i in range(len(segments) - 3):
            s1, t1, s2, t2 = segments[i], segments[i+1], segments[i+2], segments[i+3]
            
            if s1['type'] == 'S' and t1['type'] == 'T' and s2['type'] == 'S' and t2['type'] == 'T':
                if self._validate_rules(s1, t1, s2, t2):
                    return True # Geçerli bir racetrack bulundu
                    
        return False

    def _validate_rules(self, s1: Dict, t1: Dict, s2: Dict, t2: Dict) -> bool:
        """Belirlenen segmentlerin konfigürasyon kurallarına uyup uymadığını test eder."""
        
        # 1. Zıt Yönlülük Kuralı (S1 ve S2 bacakları ~180 derece zıt olmalı)
        s1_avg_heading = sum(p['heading'] for p in s1['points']) / len(s1['points'])
        s2_avg_heading = sum(p['heading'] for p in s2['points']) / len(s2['points'])
        heading_diff = abs(get_heading_diff(s1_avg_heading, s2_avg_heading))
        
        if not (180 - self.config.heading_tolerance <= heading_diff <= 180 + self.config.heading_tolerance):
            return False
            
        # 2. Süre Sınırı Kuralı
        total_time_sec = t2['end_time'] - s1['start_time']
        if total_time_sec > (self.config.max_time_min * 60):
            return False

        # 3. Kapanış/Tur Tamamlama Kuralı (Rüzgar sürüklenmesi toleransı dahil)
        start_point = s1['points'][0]
        end_point = t2['points'][-1]
        closure_dist = haversine_distance_nm(
            start_point['lat'], start_point['lon'], 
            end_point['lat'], end_point['lon']
        )
        if closure_dist > self.config.closure_tolerance_nm:
            return False

        # 4. Hava Sahası Sınırları (Bounding Box - Basitleştirilmiş Mesafe Kontrolü)
        # S1 başı ile S2 sonu arası tahmini uzunluk (Length)
        length_nm = haversine_distance_nm(
            s1['points'][0]['lat'], s1['points'][0]['lon'],
            s2['points'][-1]['lat'], s2['points'][-1]['lon']
        )
        if length_nm > self.config.max_length_nm:
            return False
            
        # S1 ile S2 arası tahmini genişlik (Width / T1 çapı)
        width_nm = haversine_distance_nm(
            t1['points'][0]['lat'], t1['points'][0]['lon'],
            t1['points'][-1]['lat'], t1['points'][-1]['lon']
        )
        if width_nm > self.config.max_width_nm:
            return False

        return True
