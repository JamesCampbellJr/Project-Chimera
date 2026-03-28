package com.projectchimera.ridelistener;

/**
 * Represents a parsed ride request from Uber or Lyft.
 */
public class RideRequest {

    public enum RideService {
        UBER, LYFT, UNKNOWN
    }

    private RideService service;
    private double totalMiles;
    private double totalPayment;
    private double riderRating;
    private int minutesToPickup;
    private int minutesToDestination;
    private String pickupAddress;
    private String destinationAddress;
    private long timestamp;

    // Calculated fields
    private int totalMinutes;
    private double pricePerHour;
    private double distanceFromCurrentLocation; // in miles

    public RideRequest() {
        this.timestamp = System.currentTimeMillis();
        this.totalMiles = -1;
        this.totalPayment = -1;
        this.riderRating = -1;
        this.minutesToPickup = -1;
        this.minutesToDestination = -1;
        this.distanceFromCurrentLocation = -1;
    }

    // --- Getters and Setters ---

    public RideService getService() { return service; }
    public void setService(RideService service) { this.service = service; }

    public double getTotalMiles() { return totalMiles; }
    public void setTotalMiles(double totalMiles) { this.totalMiles = totalMiles; }

    public double getTotalPayment() { return totalPayment; }
    public void setTotalPayment(double totalPayment) { this.totalPayment = totalPayment; }

    public double getRiderRating() { return riderRating; }
    public void setRiderRating(double riderRating) { this.riderRating = riderRating; }

    public int getMinutesToPickup() { return minutesToPickup; }
    public void setMinutesToPickup(int minutesToPickup) { this.minutesToPickup = minutesToPickup; }

    public int getMinutesToDestination() { return minutesToDestination; }
    public void setMinutesToDestination(int minutesToDestination) { this.minutesToDestination = minutesToDestination; }

    public String getPickupAddress() { return pickupAddress; }
    public void setPickupAddress(String pickupAddress) { this.pickupAddress = pickupAddress; }

    public String getDestinationAddress() { return destinationAddress; }
    public void setDestinationAddress(String destinationAddress) { this.destinationAddress = destinationAddress; }

    public long getTimestamp() { return timestamp; }

    public int getTotalMinutes() { return totalMinutes; }
    public void setTotalMinutes(int totalMinutes) { this.totalMinutes = totalMinutes; }

    public double getPricePerHour() { return pricePerHour; }
    public void setPricePerHour(double pricePerHour) { this.pricePerHour = pricePerHour; }

    public double getDistanceFromCurrentLocation() { return distanceFromCurrentLocation; }
    public void setDistanceFromCurrentLocation(double distanceFromCurrentLocation) {
        this.distanceFromCurrentLocation = distanceFromCurrentLocation;
    }

    public boolean isValid() {
        return totalMiles > 0 || totalPayment > 0;
    }

    @Override
    public String toString() {
        return "RideRequest{" +
                "service=" + service +
                ", miles=" + totalMiles +
                ", payment=$" + totalPayment +
                ", rating=" + riderRating +
                ", pickupMin=" + minutesToPickup +
                ", destMin=" + minutesToDestination +
                '}';
    }
}
