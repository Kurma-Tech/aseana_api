from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="asean")
# location = geolocator.geocode("2902B WEST TOWER")
location = geolocator.reverse("14.557871800000000, 121.023933400000000")
print(location.raw)
print((location.latitude, location.longitude))