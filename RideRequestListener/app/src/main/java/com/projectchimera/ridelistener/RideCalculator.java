package com.projectchimera.ridelistener;

/**
 * Calculates derived ride metrics from a parsed RideRequest:
 *   - Total time  = minutesToPickup + minutesToDestination
 *   - Price per hour = (totalPayment / totalMinutes) * 60
 */
public class RideCalculator {

    private RideCalculator() { /* utility class */ }

    /**
     * Populates the calculated fields of the given RideRequest in-place.
     *
     * @param request a parsed RideRequest
     */
    public static void calculate(RideRequest request) {
        if (request == null) return;

        // Total time
        int pickup = request.getMinutesToPickup();
        int dest   = request.getMinutesToDestination();

        int totalMinutes = 0;
        if (pickup > 0)  totalMinutes += pickup;
        if (dest   > 0)  totalMinutes += dest;
        request.setTotalMinutes(totalMinutes);

        // Price per hour
        double payment = request.getTotalPayment();
        if (payment > 0 && totalMinutes > 0) {
            double pricePerHour = (payment / totalMinutes) * 60.0;
            request.setPricePerHour(pricePerHour);
        } else {
            request.setPricePerHour(-1);
        }
    }

    /**
     * Returns a human-readable summary of the calculated metrics.
     */
    public static String formatSummary(RideRequest request) {
        if (request == null) return "No ride data";

        StringBuilder sb = new StringBuilder();

        if (request.getTotalMiles() > 0) {
            sb.append(String.format("Distance: %.1f mi\n", request.getTotalMiles()));
        }
        if (request.getTotalPayment() > 0) {
            sb.append(String.format("Payment: $%.2f\n", request.getTotalPayment()));
        }
        if (request.getRiderRating() > 0) {
            sb.append(String.format("Rider Rating: %.1f ★\n", request.getRiderRating()));
        }
        if (request.getMinutesToPickup() >= 0) {
            sb.append(String.format("Time to Pickup: %d min\n", request.getMinutesToPickup()));
        }
        if (request.getMinutesToDestination() >= 0) {
            sb.append(String.format("Time to Destination: %d min\n", request.getMinutesToDestination()));
        }
        if (request.getTotalMinutes() > 0) {
            sb.append(String.format("Total Trip Time: %d min\n", request.getTotalMinutes()));
        }
        if (request.getPricePerHour() > 0) {
            sb.append(String.format("Price Per Hour: $%.2f/hr\n", request.getPricePerHour()));
        }
        if (request.getDistanceFromCurrentLocation() >= 0) {
            sb.append(String.format("You are %.1f mi from pickup\n", request.getDistanceFromCurrentLocation()));
        }

        return sb.length() > 0 ? sb.toString().trim() : "Parsing in progress...";
    }
}
