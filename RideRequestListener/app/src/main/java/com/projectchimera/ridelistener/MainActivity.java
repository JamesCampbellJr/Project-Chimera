package com.projectchimera.ridelistener;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Main screen of the Ride Request Listener app.
 *
 * Shows permission status and live ride request data extracted from Uber/Lyft notifications,
 * including calculated total time, price-per-hour, and distance from current location.
 */
public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";
    private static final int LOCATION_PERMISSION_REQUEST = 1001;

    private TextView tvStatus;
    private TextView tvService;
    private TextView tvMiles;
    private TextView tvPayment;
    private TextView tvRating;
    private TextView tvPickupTime;
    private TextView tvDestTime;
    private TextView tvTotalTime;
    private TextView tvPricePerHour;
    private TextView tvDistanceToPickup;
    private TextView tvPickupAddress;
    private TextView tvRawNotif;
    private TextView tvLastUpdated;
    private View cardRideData;
    private Button btnGrantNotifPerm;
    private Button btnGrantLocationPerm;
    private Button btnOpenMaps;

    private LocationHelper locationHelper;
    private String currentPickupAddress;

    private final BroadcastReceiver rideRequestReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (!RideNotificationListenerService.ACTION_RIDE_REQUEST.equals(intent.getAction())) return;

            String serviceStr = intent.getStringExtra(RideNotificationListenerService.EXTRA_SERVICE);
            double miles      = intent.getDoubleExtra(RideNotificationListenerService.EXTRA_MILES,   -1);
            double payment    = intent.getDoubleExtra(RideNotificationListenerService.EXTRA_PAYMENT, -1);
            double rating     = intent.getDoubleExtra(RideNotificationListenerService.EXTRA_RATING,  -1);
            int pickupMin     = intent.getIntExtra(RideNotificationListenerService.EXTRA_PICKUP_MIN, -1);
            int destMin       = intent.getIntExtra(RideNotificationListenerService.EXTRA_DEST_MIN,   -1);
            String pickupAddr = intent.getStringExtra(RideNotificationListenerService.EXTRA_PICKUP_ADDR);
            String destAddr   = intent.getStringExtra(RideNotificationListenerService.EXTRA_DEST_ADDR);
            String rawText    = intent.getStringExtra(RideNotificationListenerService.EXTRA_NOTIF_TEXT);

            RideRequest request = new RideRequest();
            request.setService(parseService(serviceStr));
            request.setTotalMiles(miles);
            request.setTotalPayment(payment);
            request.setRiderRating(rating);
            request.setMinutesToPickup(pickupMin);
            request.setMinutesToDestination(destMin);
            request.setPickupAddress(pickupAddr);
            request.setDestinationAddress(destAddr);

            RideCalculator.calculate(request);
            updateRideDisplay(request, rawText);

            currentPickupAddress = pickupAddr;
            if (pickupAddr != null && !pickupAddr.isEmpty()) {
                calculateAndShowDistance(request);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        locationHelper = new LocationHelper(this);
        buildLayout();
    }

    @Override
    protected void onResume() {
        super.onResume();
        updatePermissionStatus();
        registerRideReceiver();
    }

    @Override
    protected void onPause() {
        super.onPause();
        try {
            unregisterReceiver(rideRequestReceiver);
        } catch (IllegalArgumentException ignored) { }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == LOCATION_PERMISSION_REQUEST) {
            boolean granted = grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED;
            Toast.makeText(this,
                    granted ? "Location permission granted" : "Location permission denied",
                    Toast.LENGTH_SHORT).show();
            updatePermissionStatus();
        }
    }

    // --- Layout (created programmatically to avoid external XML layout dependencies) ---

    private void buildLayout() {
        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(0xFFF5F5F5);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(16);
        root.setPadding(pad, pad, pad, pad);

        // Title
        TextView title = new TextView(this);
        title.setText("🚗 Ride Request Listener");
        title.setTextSize(22);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        title.setTextColor(0xFF1565C0);
        title.setGravity(android.view.Gravity.CENTER);
        title.setPadding(0, 0, 0, dp(8));
        root.addView(title);

        // Status
        tvStatus = new TextView(this);
        tvStatus.setText("Checking permissions…");
        tvStatus.setTextSize(14);
        tvStatus.setGravity(android.view.Gravity.CENTER);
        tvStatus.setPadding(dp(8), dp(8), dp(8), dp(8));
        tvStatus.setBackgroundColor(0xFFE8EAF6);
        LinearLayout.LayoutParams statusParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        statusParams.setMargins(0, 0, 0, dp(12));
        root.addView(tvStatus, statusParams);

        // Grant Notification Permission button
        btnGrantNotifPerm = new Button(this);
        btnGrantNotifPerm.setText("Enable Notification Access");
        btnGrantNotifPerm.setBackgroundColor(0xFF000000);
        btnGrantNotifPerm.setTextColor(0xFFFFFFFF);
        btnGrantNotifPerm.setVisibility(View.GONE);
        LinearLayout.LayoutParams btnParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        btnParams.setMargins(0, 0, 0, dp(8));
        root.addView(btnGrantNotifPerm, btnParams);
        btnGrantNotifPerm.setOnClickListener(v -> openNotificationListenerSettings());

        // Grant Location Permission button
        btnGrantLocationPerm = new Button(this);
        btnGrantLocationPerm.setText("Grant Location Permission");
        btnGrantLocationPerm.setBackgroundColor(0xFF1565C0);
        btnGrantLocationPerm.setTextColor(0xFFFFFFFF);
        btnGrantLocationPerm.setVisibility(View.GONE);
        LinearLayout.LayoutParams btnLocParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        btnLocParams.setMargins(0, 0, 0, dp(12));
        root.addView(btnGrantLocationPerm, btnLocParams);
        btnGrantLocationPerm.setOnClickListener(v -> requestLocationPermission());

        // Ride Data card (initially hidden)
        cardRideData = buildRideCard(root);
        cardRideData.setVisibility(View.GONE);

        // Waiting message
        TextView tvWaiting = new TextView(this);
        tvWaiting.setText("Waiting for a ride request notification…\n\nMake sure the Uber or Lyft driver app is installed and notification access is enabled.");
        tvWaiting.setTextSize(15);
        tvWaiting.setGravity(android.view.Gravity.CENTER);
        tvWaiting.setTextColor(0xFF757575);
        tvWaiting.setPadding(dp(16), dp(32), dp(16), dp(32));
        root.addView(tvWaiting);

        scroll.addView(root);
        setContentView(scroll);
    }

    private View buildRideCard(LinearLayout parent) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setBackgroundColor(0xFFFFFFFF);
        int cardPad = dp(16);
        card.setPadding(cardPad, cardPad, cardPad, cardPad);

        LinearLayout.LayoutParams cardParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        cardParams.setMargins(0, 0, 0, dp(12));

        tvService = addLabel(card, "Ride Service", true, 0xFF1565C0, 18);
        addDivider(card);

        addSectionHeader(card, "Ride Information");
        tvMiles    = addLabel(card, "Miles", false, 0xFF212121, 15);
        tvPayment  = addLabel(card, "Payment", false, 0xFF212121, 15);
        tvRating   = addLabel(card, "Rider Rating", false, 0xFF212121, 15);
        tvPickupTime = addLabel(card, "Time to Pickup", false, 0xFF212121, 15);
        tvDestTime = addLabel(card, "Time to Destination", false, 0xFF212121, 15);

        addDivider(card);
        addSectionHeader(card, "Calculated Metrics");
        tvTotalTime    = addLabel(card, "Total Time", true, 0xFF212121, 15);
        tvPricePerHour = addLabel(card, "Price/Hour", true, 0xFF212121, 15);

        addDivider(card);
        addSectionHeader(card, "Location");
        tvDistanceToPickup = addLabel(card, "Distance to Pickup", true, 0xFFFF6F00, 15);
        tvPickupAddress    = addLabel(card, "Pickup Address", false, 0xFF757575, 13);

        btnOpenMaps = new Button(this);
        btnOpenMaps.setText("Open Pickup in Maps");
        btnOpenMaps.setBackgroundColor(0xFF1565C0);
        btnOpenMaps.setTextColor(0xFFFFFFFF);
        btnOpenMaps.setVisibility(View.GONE);
        LinearLayout.LayoutParams mapsParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        mapsParams.setMargins(0, 0, 0, dp(8));
        card.addView(btnOpenMaps, mapsParams);
        btnOpenMaps.setOnClickListener(v -> openMapsForPickup());

        addDivider(card);

        tvRawNotif = new TextView(this);
        tvRawNotif.setTextSize(11);
        tvRawNotif.setTextColor(0xFF757575);
        tvRawNotif.setVisibility(View.GONE);
        card.addView(tvRawNotif);

        tvLastUpdated = new TextView(this);
        tvLastUpdated.setTextSize(11);
        tvLastUpdated.setTextColor(0xFF757575);
        tvLastUpdated.setGravity(android.view.Gravity.END);
        card.addView(tvLastUpdated);

        parent.addView(card, cardParams);
        return card;
    }

    private TextView addLabel(LinearLayout parent, String text, boolean bold, int color, float size) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextSize(size);
        tv.setTextColor(color);
        if (bold) tv.setTypeface(null, android.graphics.Typeface.BOLD);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        p.setMargins(0, 0, 0, dp(4));
        parent.addView(tv, p);
        return tv;
    }

    private void addSectionHeader(LinearLayout parent, String text) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextSize(13);
        tv.setTextColor(0xFF455A64);
        tv.setTypeface(null, android.graphics.Typeface.BOLD);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        p.setMargins(0, 0, 0, dp(6));
        parent.addView(tv, p);
    }

    private void addDivider(LinearLayout parent) {
        View div = new View(this);
        div.setBackgroundColor(0xFFE0E0E0);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 1);
        p.setMargins(0, dp(12), 0, dp(12));
        parent.addView(div, p);
    }

    private int dp(int dp) {
        return (int) (dp * getResources().getDisplayMetrics().density + 0.5f);
    }

    // --- Permission handling ---

    private void updatePermissionStatus() {
        boolean notifEnabled = isNotificationListenerEnabled();
        boolean locationOk   = locationHelper.hasLocationPermission();

        if (notifEnabled && locationOk) {
            tvStatus.setText("✅ Ready — Listening for Uber & Lyft requests");
            tvStatus.setTextColor(0xFF2E7D32);
            btnGrantNotifPerm.setVisibility(View.GONE);
            btnGrantLocationPerm.setVisibility(View.GONE);
        } else {
            if (!notifEnabled) {
                tvStatus.setText("⚠️ Notification listener permission required");
                tvStatus.setTextColor(0xFFC62828);
                btnGrantNotifPerm.setVisibility(View.VISIBLE);
            } else {
                btnGrantNotifPerm.setVisibility(View.GONE);
            }
            if (!locationOk) {
                btnGrantLocationPerm.setVisibility(View.VISIBLE);
                if (notifEnabled) {
                    tvStatus.setText("⚠️ Location permission required for distance calculation");
                    tvStatus.setTextColor(0xFFE65100);
                }
            } else {
                btnGrantLocationPerm.setVisibility(View.GONE);
            }
        }
    }

    private boolean isNotificationListenerEnabled() {
        String flat = Settings.Secure.getString(getContentResolver(), "enabled_notification_listeners");
        if (flat == null || flat.isEmpty()) return false;
        ComponentName myComponent = new ComponentName(this, RideNotificationListenerService.class);
        for (String pkg : flat.split(":")) {
            ComponentName cn = ComponentName.unflattenFromString(pkg);
            if (myComponent.equals(cn)) return true;
        }
        return false;
    }

    private void openNotificationListenerSettings() {
        new AlertDialog.Builder(this)
                .setTitle("Notification Access Required")
                .setMessage("This app needs Notification Access to read Uber and Lyft ride requests. " +
                        "Please enable it in the next screen by finding \"Ride Request Listener\" and toggling it on.")
                .setPositiveButton("Open Settings", (d, w) ->
                        startActivity(new Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)))
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void requestLocationPermission() {
        requestPermissions(new String[]{
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
        }, LOCATION_PERMISSION_REQUEST);
    }

    private void registerRideReceiver() {
        IntentFilter filter = new IntentFilter(RideNotificationListenerService.ACTION_RIDE_REQUEST);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(rideRequestReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(rideRequestReceiver, filter);
        }
    }

    // --- UI Updates ---

    private void updateRideDisplay(RideRequest request, String rawText) {
        cardRideData.setVisibility(View.VISIBLE);

        String svcName = request.getService() != null ? request.getService().name() : "UNKNOWN";
        tvService.setText("Service: " + svcName);

        tvMiles.setText(request.getTotalMiles() > 0
                ? String.format("Total Miles: %.1f mi", request.getTotalMiles()) : "Total Miles: N/A");

        tvPayment.setText(request.getTotalPayment() > 0
                ? String.format("Payment: $%.2f", request.getTotalPayment()) : "Payment: N/A");

        tvRating.setText(request.getRiderRating() > 0
                ? String.format("Rider Rating: %.1f ★", request.getRiderRating()) : "Rider Rating: N/A");

        tvPickupTime.setText(request.getMinutesToPickup() >= 0
                ? String.format("Time to Pickup: %d min", request.getMinutesToPickup())
                : "Time to Pickup: N/A");

        tvDestTime.setText(request.getMinutesToDestination() >= 0
                ? String.format("Time to Destination: %d min", request.getMinutesToDestination())
                : "Time to Destination: N/A");

        tvTotalTime.setText(request.getTotalMinutes() > 0
                ? String.format("⏱ Total Trip Time: %d min", request.getTotalMinutes())
                : "⏱ Total Trip Time: N/A");

        tvPricePerHour.setText(request.getPricePerHour() > 0
                ? String.format("💰 Price Per Hour: $%.2f/hr", request.getPricePerHour())
                : "💰 Price Per Hour: N/A");

        tvDistanceToPickup.setText("📍 Calculating distance…");

        String addr = request.getPickupAddress();
        tvPickupAddress.setText(addr != null && !addr.isEmpty()
                ? "Pickup: " + addr : "Pickup address not found in notification");

        if (rawText != null && !rawText.trim().isEmpty()) {
            tvRawNotif.setText("Raw: " + rawText.trim());
            tvRawNotif.setVisibility(View.VISIBLE);
        } else {
            tvRawNotif.setVisibility(View.GONE);
        }

        String time = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
        tvLastUpdated.setText("Updated: " + time);

        btnOpenMaps.setVisibility(addr != null && !addr.isEmpty() ? View.VISIBLE : View.GONE);
    }

    private void calculateAndShowDistance(RideRequest request) {
        locationHelper.calculateDistanceToPickup(request.getPickupAddress(),
                new LocationHelper.DistanceCallback() {
                    @Override
                    public void onDistance(double distanceMiles) {
                        request.setDistanceFromCurrentLocation(distanceMiles);
                        runOnUiThread(() -> tvDistanceToPickup.setText(
                                String.format("📍 Distance to Pickup: %.1f mi away", distanceMiles)));
                    }

                    @Override
                    public void onError(String error) {
                        Log.w(TAG, "Distance error: " + error);
                        runOnUiThread(() -> tvDistanceToPickup.setText("📍 Distance: " + error));
                    }
                });
    }

    private void openMapsForPickup() {
        if (currentPickupAddress == null || currentPickupAddress.isEmpty()) {
            Toast.makeText(this, "No pickup address available", Toast.LENGTH_SHORT).show();
            return;
        }
        String encoded = Uri.encode(currentPickupAddress);
        Uri gmmUri = Uri.parse("geo:0,0?q=" + encoded);
        Intent mapIntent = new Intent(Intent.ACTION_VIEW, gmmUri);
        mapIntent.setPackage("com.google.android.apps.maps");
        if (mapIntent.resolveActivity(getPackageManager()) != null) {
            startActivity(mapIntent);
        } else {
            Uri webUri = Uri.parse("https://maps.google.com/?q=" + encoded);
            startActivity(new Intent(Intent.ACTION_VIEW, webUri));
        }
    }

    private RideRequest.RideService parseService(String name) {
        if (name == null) return RideRequest.RideService.UNKNOWN;
        try {
            return RideRequest.RideService.valueOf(name);
        } catch (IllegalArgumentException e) {
            return RideRequest.RideService.UNKNOWN;
        }
    }
}
