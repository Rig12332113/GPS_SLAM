#include <iostream>
#include "GPSsample.hpp"

void GPSsample::setTime(double t){
    timestamp = t;
} 

void GPSsample::setLatitude(double lat){
    latitude = lat;
} 

void GPSsample::setLongitude(double lon){
    longitude = lon;
}

void GPSsample::setAltitude(double alt){
    altitude = alt;
}

void GPSsample::setHAcc(double hAcc){
    hacc = hAcc;
}

void GPSsample::setVAcc(double vAcc){
    vacc = vAcc;
}

void GPSsample::setSpeed(double spe){
    speed = spe;
}

void GPSsample::setCourse(double cour){
    course = cour;
}

void GPSsample::setTGPS(double tGPS){
    t_gps = tGPS;
}   

double GPSsample::getTimestamp() const{
    return timestamp;
}

double GPSsample::getLatitude() const{
    return latitude;
}

double GPSsample::getLongitude() const{
    return longitude;
}

double GPSsample::getHAcc() const { 
    return hacc; 
}

std::ostream& operator<<(std::ostream& os, const GPSsample& sample){
    os << "GPS data: ";
    os << "t: " << sample.timestamp << "\n";
    os << "Latitude: " << sample.latitude << " Longitude: " << sample.longitude << " Altitude: " << sample .altitude << "\n";
    os << "hAcc: " << sample.hacc << " vAcc: " << sample.vacc << " speed: " << sample.speed << " t_gps: " << sample.t_gps << "\n";
    return os;
}