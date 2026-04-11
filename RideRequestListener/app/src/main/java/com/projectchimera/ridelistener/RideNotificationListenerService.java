package com.projectchimera.ridelistener;

import android.service.notification.NotificationListenerService;
import android.service.notification.StatusBarNotification;
import android.app.Notification;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;

/**
 * Listens for notifications from Uber and Lyft apps, parses ride request data,
 * and broadcasts it to the MainActivity.
 */
public class RideNotificationListenerService extends NotificationListenerService {

    private static final String TAG = "RideNotifListener";

    // Package names for Uber and Lyft
    static final String UBER_PACKAGE = "com.ubercab";
    static final String UBER_DRIVER_PACKAGE = "com.ubercab.driver";
    static final String LYFT_PACKAGE = "com.lyft.android";
    static final String LYFT_DRIVER_PACKAGE = "com.lyft.android.driver";

    public static final String ACTION_RIDE_REQUEST = "com.projectchimera.ridelistener.RIDE_REQUEST";
    public static final String EXTRA_RIDE_REQUEST = "ride_request_data";

    // Extras broadcast to MainActivity
    public static final String EXTRA_SERVICE       = "service";
    public static final String EXTRA_MILES         = "miles";
    public static final String EXTRA_PAYMENT       = "payment";
    public static final String EXTRA_RATING        = "rating";
    public static final String EXTRA_PICKUP_MIN    = "pickup_minutes";
    public static final String EXTRA_DEST_MIN      = "dest_minutes";
    public static final String EXTRA_PICKUP_ADDR   = "pickup_address";
    public static final String EXTRA_DEST_ADDR     = "dest_address";
    public static final String EXTRA_NOTIF_TEXT    = "notification_text";

    @Override
    public void onNotificationPosted(StatusBarNotification sbn) {
        if (sbn == null) return;

        String pkg = sbn.getPackageName();
        if (!isRideAppPackage(pkg)) return;

        Log.d(TAG, "Notification from: " + pkg);

        Notification notification = sbn.getNotification();
        if (notification == null) return;

        Bundle extras = notification.extras;
        if (extras == null) return;

        String title = extras.getString(Notification.EXTRA_TITLE, "");
        String text  = extras.getString(Notification.EXTRA_TEXT, "");
        String bigText = extras.getString(Notification.EXTRA_BIG_TEXT, "");

        // Combine all text sources for parsing
        String fullText = title + " " + text + " " + bigText;
        Log.d(TAG, "Full notification text: " + fullText);

        // Determine which service
        RideRequest.RideService service = getServiceFromPackage(pkg);

        // Parse the notification
        RideRequest request = RideInfoParser.parse(fullText, service);

        if (request != null && request.isValid()) {
            Log.d(TAG, "Parsed ride request: " + request);
            broadcastRideRequest(request, fullText);
        } else {
            Log.d(TAG, "Notification did not contain a valid ride request, raw text forwarded");
            // Forward raw text so the UI can still show it
            broadcastRawNotification(service, fullText);
        }
    }

    @Override
    public void onNotificationRemoved(StatusBarNotification sbn) {
        // Not needed for this use case
    }

    private boolean isRideAppPackage(String pkg) {
        return UBER_PACKAGE.equals(pkg)
                || UBER_DRIVER_PACKAGE.equals(pkg)
                || LYFT_PACKAGE.equals(pkg)
                || LYFT_DRIVER_PACKAGE.equals(pkg);
    }

    private RideRequest.RideService getServiceFromPackage(String pkg) {
        if (UBER_PACKAGE.equals(pkg) || UBER_DRIVER_PACKAGE.equals(pkg)) {
            return RideRequest.RideService.UBER;
        } else if (LYFT_PACKAGE.equals(pkg) || LYFT_DRIVER_PACKAGE.equals(pkg)) {
            return RideRequest.RideService.LYFT;
        }
        return RideRequest.RideService.UNKNOWN;
    }

    private void broadcastRideRequest(RideRequest request, String rawText) {
        Intent intent = new Intent(ACTION_RIDE_REQUEST);
        intent.setPackage(getPackageName());
        intent.putExtra(EXTRA_SERVICE,      request.getService().name());
        intent.putExtra(EXTRA_MILES,        request.getTotalMiles());
        intent.putExtra(EXTRA_PAYMENT,      request.getTotalPayment());
        intent.putExtra(EXTRA_RATING,       request.getRiderRating());
        intent.putExtra(EXTRA_PICKUP_MIN,   request.getMinutesToPickup());
        intent.putExtra(EXTRA_DEST_MIN,     request.getMinutesToDestination());
        intent.putExtra(EXTRA_PICKUP_ADDR,  request.getPickupAddress());
        intent.putExtra(EXTRA_DEST_ADDR,    request.getDestinationAddress());
        intent.putExtra(EXTRA_NOTIF_TEXT,   rawText);
        sendBroadcast(intent);
    }

    private void broadcastRawNotification(RideRequest.RideService service, String rawText) {
        Intent intent = new Intent(ACTION_RIDE_REQUEST);
        intent.setPackage(getPackageName());
        intent.putExtra(EXTRA_SERVICE,    service.name());
        intent.putExtra(EXTRA_NOTIF_TEXT, rawText);
        intent.putExtra(EXTRA_MILES,      -1.0);
        intent.putExtra(EXTRA_PAYMENT,    -1.0);
        intent.putExtra(EXTRA_RATING,     -1.0);
        intent.putExtra(EXTRA_PICKUP_MIN, -1);
        intent.putExtra(EXTRA_DEST_MIN,   -1);
        sendBroadcast(intent);
    }
}
