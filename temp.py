import math

class CAPOrbitDetector:
    def __init__(self, config=None):
        # --- Konfigürasyon (İhtiyaca göre dışarıdan güncellenebilir) ---
        self.config = config or {
            'min_turn_rate': 1.2,        # 10 sn'de minimum dönüş eşiği (Gürültü filtresi)
            'target_lap_angle': 345.0,   # Tam tur için birikmesi gereken mutlak açı
            'dist_tolerance_km': 7.5,    # Başlangıç noktasına dönüş toleransı (Orbit çapına göre)
            'alt_tolerance_m': 300,      # Tur boyunca izin verilen max irtifa değişimi (metre)
            'max_orbit_time_sec': 900,   # Bir turun tamamlanması için max süre (15 dk)
            'straight_limit_sec': 180    # Manevra içinde max düz uçuş süresi (3 dk)
        }
        
        self.completed_laps = []
        self.reset_state()

    def reset_state(self):
        """Tüm geçici takip değişkenlerini sıfırlar."""
        self.last_telemetry = None
        self.lap_start_point = None
        self.accumulated_angle = 0.0
        self.min_alt = float('inf')
        self.max_alt = float('-inf')
        self.in_maneuver = False
        self.start_time = None
        self.last_turn_time = None

    def _get_heading_diff(self, h1, h2):
        """İki açı arasındaki en kısa farkı hesaplar (-180, 180)."""
        return ((h2 - h1 + 180) % 360) - 180

    def _get_distance_km(self, p1, p2):
        """Haversine formülü ile mesafe ölçümü."""
        R = 6371.0
        phi1, phi2 = math.radians(p1['lat']), math.radians(p2['lat'])
        d_phi = math.radians(p2['lat'] - p1['lat'])
        d_lambda = math.radians(p2['lon'] - p1['lon'])
        
        a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def update(self, current_pt):
        """
        Her 10 saniyede bir gelen veriyi işler.
        current_pt: {'time': unix_timestamp, 'lat': float, 'lon': float, 'heading': float, 'alt': float}
        """
        # İlk veri kontrolü
        if self.last_telemetry is None:
            self.last_telemetry = current_pt
            return None

        # 1. Anlık değişimleri hesapla
        delta_h = self._get_heading_diff(self.last_telemetry['heading'], current_pt['heading'])
        
        # 2. Manevra Durum Kontrolü
        if not self.in_maneuver:
            # Uçak anlamlı bir dönüşe başladı mı?
            if abs(delta_h) >= self.config['min_turn_rate']:
                self.in_maneuver = True
                self.lap_start_point = self.last_telemetry
                self.start_time = current_pt['time']
                self.last_turn_time = current_pt['time']
                self.accumulated_angle = delta_h
                self.min_alt = min(self.last_telemetry['alt'], current_pt['alt'])
                self.max_alt = max(self.last_telemetry['alt'], current_pt['alt'])
        else:
            # Manevra içerisindeyiz, verileri biriktir
            self.accumulated_angle += delta_h
            self.min_alt = min(self.min_alt, current_pt['alt'])
            self.max_alt = max(self.max_alt, current_pt['alt'])
            
            if abs(delta_h) >= self.config['min_turn_rate']:
                self.last_turn_time = current_pt['time']

            # --- GÜVENLİK FİLTRELERİ (Edge Case Yönetimi) ---
            
            # A. Zaman Aşımı: Tur çok uzun sürdüyse (örn. 15 dk) sıfırla.
            duration = current_pt['time'] - self.start_time
            if duration > self.config['max_orbit_time_sec']:
                self.reset_state()
                return None

            # B. Düz Uçuş Kopması: Manevra içinde çok uzun süre (3 dk) düz gittiyse CAP değildir.
            if (current_pt['time'] - self.last_turn_time) > self.config['straight_limit_sec']:
                self.reset_state()
                return None

            # 3. TUR TAMAMLANMA KOŞULU (Final Check)
            if abs(self.accumulated_angle) >= self.config['target_lap_angle']:
                dist = self._get_distance_km(self.lap_start_point, current_pt)
                alt_diff = self.max_alt - self.min_alt
                
                # Koşullar: Mesafe yakın mı? İrtifa stabil mi?
                if dist <= self.config['dist_tolerance_km'] and alt_diff <= self.config['alt_tolerance_m']:
                    # BAŞARILI TUR TESPİTİ
                    lap_report = {
                        'lap_index': len(self.completed_laps) + 1,
                        'start_time': self.start_time,
                        'end_time': current_pt['time'],
                        'duration_sec': duration,
                        'avg_alt': (self.max_alt + self.min_alt) / 2,
                        'alt_deviation': alt_diff,
                        'direction': 'Right' if self.accumulated_angle > 0 else 'Left',
                        'start_pos': (self.lap_start_point['lat'], self.lap_start_point['lon']),
                        'end_pos': (current_pt['lat'], current_pt['lon'])
                    }
                    self.completed_laps.append(lap_report)
                    
                    # Sonraki tur için akışı bozmadan resetle
                    self.reset_state()
                    # Mevcut nokta yeni turun potansiyel başlangıcıdır
                    self.in_maneuver = True
                    self.lap_start_point = current_pt
                    self.start_time = current_pt['time']
                    self.last_turn_time = current_pt['time']
                    
                    return lap_report

        self.last_telemetry = current_pt
        return None
