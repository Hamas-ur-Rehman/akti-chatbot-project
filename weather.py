import os
import requests
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

load_dotenv()

def get_lat_long(location_name):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(location_name)
    return (location.latitude, location.longitude) if location else (None, None)

def get_weather_data(location_name, exclude="minutely,hourly", units="metric"):
    lat, lon = get_lat_long(location_name)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
    
    response = requests.get(url)
    if response.status_code == 200:
        response = response.json()
        pretty_response = f"""Current temperature in {location_name} is {response['current']['temperature_2m']}°C.
        """
        return pretty_response
    else:
        response.raise_for_status()

