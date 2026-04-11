package com.projectchimera.ridelistener;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses Uber and Lyft notification text to extract ride request data.
 *
 * Uber driver notifications typically contain patterns like:
 *   "New trip request | 3.2 mi | $12.50 | 4.8★ | 4 min away | 18 min trip"
 *
 * Lyft driver notifications typically contain patterns like:
 *   "New ride request | 2.5 miles | $9.75 | Rating: 4.9 | Pickup in 6 min | 22 min to destination"
 *
 * Because notification formats change over time, the parser uses flexible regex patterns
 * that can match a wide variety of text structures.
 */
public class RideInfoParser {

    // --- Distance patterns ---
    // Matches: "3.2 mi", "3.2 miles", "3.2mi"
    private static final Pattern MILES_PATTERN = Pattern.compile(
            "(\\d+(?:\\.\\d+)?)\\s*mi(?:les?)?\\b", Pattern.CASE_INSENSITIVE);

    // --- Payment patterns ---
    // Matches: "$12.50", "12.50 USD", "earn $12.50", "guaranteed $8"
    private static final Pattern PAYMENT_PATTERN = Pattern.compile(
            "\\$\\s*(\\d+(?:\\.\\d+)?)", Pattern.CASE_INSENSITIVE);

    // --- Rating patterns ---
    // Matches: "4.8★", "4.8 stars", "Rating: 4.9", "rated 4.7", "4.8 rating"
    private static final Pattern RATING_PATTERN = Pattern.compile(
            "(?:rating[:\\s]+|rated\\s+)?(\\d+(?:\\.\\d+)?)\\s*(?:★|stars?|rating)",
            Pattern.CASE_INSENSITIVE);

    // --- Pickup time patterns ---
    // Matches: "4 min away", "4 min pickup", "pickup in 4 min", "4 minutes away"
    private static final Pattern PICKUP_TIME_PATTERN = Pattern.compile(
            "(?:pickup\\s+(?:in\\s+)?)?(\\d+)\\s*min(?:utes?)?\\s*away|" +
            "(?:pickup\\s+in\\s+)(\\d+)\\s*min(?:utes?)?|" +
            "(\\d+)\\s*min(?:utes?)?\\s+(?:to\\s+)?pickup",
            Pattern.CASE_INSENSITIVE);

    // --- Destination / trip time patterns ---
    // Matches: "18 min trip", "18 min to destination", "trip: 18 min", "22 min ride"
    private static final Pattern DEST_TIME_PATTERN = Pattern.compile(
            "(\\d+)\\s*min(?:utes?)?\\s*(?:trip|ride|to\\s+dest(?:ination)?)|" +
            "(?:trip|ride)[:\\s]+(\\d+)\\s*min(?:utes?)?",
            Pattern.CASE_INSENSITIVE);

    // --- Address patterns ---
    // Matches lines that look like street addresses (number + street name)
    private static final Pattern ADDRESS_PATTERN = Pattern.compile(
            "\\d+\\s+[A-Za-z][A-Za-z0-9\\s,\\.]+(?:St|Ave|Blvd|Dr|Rd|Ln|Way|Ct|Pl|Pkwy)\\.?",
            Pattern.CASE_INSENSITIVE);

    private RideInfoParser() { /* utility class */ }

    /**
     * Parses notification text and returns a populated RideRequest, or null if nothing useful
     * was found.
     */
    public static RideRequest parse(String text, RideRequest.RideService service) {
        if (text == null || text.trim().isEmpty()) return null;

        RideRequest request = new RideRequest();
        request.setService(service);

        // Miles
        Matcher milesMatcher = MILES_PATTERN.matcher(text);
        if (milesMatcher.find()) {
            request.setTotalMiles(Double.parseDouble(milesMatcher.group(1)));
        }

        // Payment
        Matcher paymentMatcher = PAYMENT_PATTERN.matcher(text);
        if (paymentMatcher.find()) {
            request.setTotalPayment(Double.parseDouble(paymentMatcher.group(1)));
        }

        // Rating
        Matcher ratingMatcher = RATING_PATTERN.matcher(text);
        if (ratingMatcher.find()) {
            String ratingStr = ratingMatcher.group(1);
            if (ratingStr != null) {
                double rating = Double.parseDouble(ratingStr);
                // Ratings are between 1.0 and 5.0
                if (rating >= 1.0 && rating <= 5.0) {
                    request.setRiderRating(rating);
                }
            }
        }

        // Pickup time
        Matcher pickupMatcher = PICKUP_TIME_PATTERN.matcher(text);
        if (pickupMatcher.find()) {
            String mins = firstNonNull(pickupMatcher.group(1), pickupMatcher.group(2), pickupMatcher.group(3));
            if (mins != null) {
                request.setMinutesToPickup(Integer.parseInt(mins));
            }
        }

        // Destination / trip time
        Matcher destMatcher = DEST_TIME_PATTERN.matcher(text);
        if (destMatcher.find()) {
            String mins = firstNonNull(destMatcher.group(1), destMatcher.group(2));
            if (mins != null) {
                request.setMinutesToDestination(Integer.parseInt(mins));
            }
        }

        // Try to extract addresses
        Matcher addrMatcher = ADDRESS_PATTERN.matcher(text);
        if (addrMatcher.find()) {
            request.setPickupAddress(addrMatcher.group().trim());
            if (addrMatcher.find()) {
                request.setDestinationAddress(addrMatcher.group().trim());
            }
        }

        return request;
    }

    private static String firstNonNull(String... values) {
        for (String v : values) {
            if (v != null) return v;
        }
        return null;
    }
}
