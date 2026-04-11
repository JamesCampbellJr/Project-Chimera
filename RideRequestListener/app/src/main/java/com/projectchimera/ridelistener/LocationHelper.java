package com.projectchimera.ridelistener;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.location.Address;
import android.location.Geocoder;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.util.Log;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

/**
 * Helper for location operations using the standard Android LocationManager.
 * No Play Services dependency.
 */
public class LocationHelper {

    private static final String TAG = "LocationHelper";
    private static final double METERS_PER_MILE = 1609.344;

    public interface LocationCallback {
        void onLocation(double latitude, double longitude);
        void onError(String error);
    }

    public interface DistanceCallback {
        void onDistance(double distanceMiles);
        void onError(String error);
    }

    private final Context context;
    private final LocationManager locationManager;

    public LocationHelper(Context context) {
        this.context = context.getApplicationContext();
        this.locationManager = (LocationManager) this.context.getSystemService(Context.LOCATION_SERVICE);
    }

    public boolean hasLocationPermission() {
        return context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED
                || context.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION)
                == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Gets the device's current (or last-known) location asynchronously.
     */
    public void getCurrentLocation(LocationCallback callback) {
        if (!hasLocationPermission()) {
            callback.onError("Location permission not granted");
            return;
        }

        try {
            // Try GPS first, fall back to network
            Location gps = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
            Location network = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
            Location best = chooseBest(gps, network);

            if (best != null) {
                callback.onLocation(best.getLatitude(), best.getLongitude());
            } else {
                // Request a single update
                requestSingleUpdate(callback);
            }
        } catch (SecurityException e) {
            callback.onError("Location permission denied: " + e.getMessage());
        }
    }

    private void requestSingleUpdate(LocationCallback callback) {
        String provider = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
                ? LocationManager.GPS_PROVIDER
                : LocationManager.NETWORK_PROVIDER;

        try {
            locationManager.requestSingleUpdate(provider, new LocationListener() {
                @Override
                public void onLocationChanged(Location location) {
                    callback.onLocation(location.getLatitude(), location.getLongitude());
                }
                @Override
                public void onStatusChanged(String p, int st, Bundle extras) {}
                @Override
                public void onProviderEnabled(String p) {}
                @Override
                public void onProviderDisabled(String p) {
                    callback.onError("Location provider disabled");
                }
            }, null);
        } catch (SecurityException e) {
            callback.onError("Location permission denied: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            callback.onError("No location provider available");
        }
    }

    private Location chooseBest(Location a, Location b) {
        if (a == null) return b;
        if (b == null) return a;
        return a.getTime() >= b.getTime() ? a : b;
    }

    /**
     * Geocodes an address to lat/lng. Must be called off the main thread.
     */
    public double[] geocodeAddress(String address) {
        if (address == null || address.trim().isEmpty()) return null;
        if (!Geocoder.isPresent()) return null;

        Geocoder geocoder = new Geocoder(context, Locale.getDefault());
        try {
            List<Address> results = geocoder.getFromLocationName(address, 1);
            if (results != null && !results.isEmpty()) {
                Address addr = results.get(0);
                return new double[]{addr.getLatitude(), addr.getLongitude()};
            }
        } catch (IOException e) {
            Log.e(TAG, "Geocoding failed for: " + address, e);
        }
        return null;
    }

    public static double distanceMiles(double lat1, double lon1, double lat2, double lon2) {
        float[] results = new float[1];
        Location.distanceBetween(lat1, lon1, lat2, lon2, results);
        return results[0] / METERS_PER_MILE;
    }

    public void calculateDistanceToPickup(String pickupAddress, DistanceCallback callback) {
        if (pickupAddress == null || pickupAddress.trim().isEmpty()) {
            callback.onError("No pickup address provided");
            return;
        }

        getCurrentLocation(new LocationCallback() {
            @Override
            public void onLocation(double userLat, double userLon) {
                new Thread(() -> {
                    double[] coords = geocodeAddress(pickupAddress);
                    if (coords == null) {
                        callback.onError("Could not geocode: " + pickupAddress);
                        return;
                    }
                    double dist = distanceMiles(userLat, userLon, coords[0], coords[1]);
                    callback.onDistance(dist);
                }).start();
            }

            @Override
            public void onError(String error) {
                callback.onError(error);
            }
        });
    }
}
