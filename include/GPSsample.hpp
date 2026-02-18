class GPSsample{
public:
    GPSsample() = default;

    void setTime(double t);
    void setLatitude(double lat);
    void setLongitude(double lon);
    void setAltitude(double alt);
    void setHAcc(double hAcc);
    void setVAcc(double vAcc);
    void setSpeed(double spe);
    // coarse is direction of travel, from 0 to 360 degrees counted clockwise from true north.
    // Value -1 indicate invalid
    void setCourse(double course);
    // t_gps is the time from gps subsystem
    void setTGPS(double t_gps);

    double getTimestamp() const;
    double getLatitude() const;
    double getLongitude() const;
    double getHAcc() const;
    friend std::ostream& operator<<(std::ostream& os, const GPSsample& sample);

private:
    double timestamp;
    double latitude;
    double longitude;
    double altitude;
    double hacc;
    double vacc;
    double speed;
    double course;
    double t_gps;
};