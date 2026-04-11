#!/bin/bash
set -e

ANDROID_HOME="${ANDROID_HOME:-/usr/local/lib/android/sdk}"
BUILD_TOOLS="$ANDROID_HOME/build-tools/35.0.0"
PLATFORM="$ANDROID_HOME/platforms/android-35"
ANDROID_JAR="$PLATFORM/android.jar"

PROJECT_DIR="/home/runner/work/Project-Chimera/Project-Chimera/RideRequestListener"
APP_DIR="$PROJECT_DIR/app"
SRC_DIR="$APP_DIR/src/main/java"
RES_DIR="$APP_DIR/src/main/res"
MANIFEST="$APP_DIR/src/main/AndroidManifest.xml"
BUILD_DIR="/tmp/ride_listener_build"
OUT_APK="$PROJECT_DIR/RideRequestListener-debug.apk"

AAPT2="$BUILD_TOOLS/aapt2"
D8="$BUILD_TOOLS/d8"
APKSIGNER="$BUILD_TOOLS/apksigner"
ZIPALIGN="$BUILD_TOOLS/zipalign"

echo "=== Build dirs ==="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/obj/com/projectchimera/ridelistener"
mkdir -p "$BUILD_DIR/dex"
mkdir -p "$BUILD_DIR/res_compiled"

echo "=== Step 1: Compile resources with aapt2 ==="
find "$RES_DIR" -type f -name "*.xml" | while read f; do
    $AAPT2 compile "$f" -o "$BUILD_DIR/res_compiled/" 2>&1 || echo "Warning: failed $f"
done
echo "Resources compiled:"
ls "$BUILD_DIR/res_compiled/"

echo "=== Step 2: Link resources (positional flat files, not -R) ==="
# Pass flat files as positional arguments (not -R overlays)
FLAT_FILES=$(find "$BUILD_DIR/res_compiled" -name "*.flat" | tr '\n' ' ')

$AAPT2 link \
    -o "$BUILD_DIR/unsigned.apk" \
    --manifest "$MANIFEST" \
    -I "$ANDROID_JAR" \
    --java "$BUILD_DIR/obj" \
    --min-sdk-version 26 \
    --target-sdk-version 35 \
    --version-code 1 \
    --version-name "1.0" \
    $FLAT_FILES \
    2>&1
echo "Resources linked"
echo "Generated files in obj:"
find "$BUILD_DIR/obj" -name "*.java"

echo "=== Step 3: Compile Java sources ==="
JAVA_FILES=$(find "$SRC_DIR" -name "*.java" | tr '\n' ' ')
R_JAVA=$(find "$BUILD_DIR/obj" -name "*.java" | tr '\n' ' ')

javac \
    --release 8 \
    -cp "$ANDROID_JAR" \
    -d "$BUILD_DIR/obj" \
    $JAVA_FILES $R_JAVA \
    2>&1
echo "Java compiled"

echo "=== Step 4: Convert to DEX ==="
CLASS_FILES=$(find "$BUILD_DIR/obj" -name "*.class" | tr '\n' ' ')
$D8 \
    --output "$BUILD_DIR/dex" \
    --lib "$ANDROID_JAR" \
    --min-api 26 \
    $CLASS_FILES \
    2>&1
echo "DEX compiled"

echo "=== Step 5: Add DEX to APK ==="
cp "$BUILD_DIR/dex/classes.dex" "$BUILD_DIR/"
cd "$BUILD_DIR"
zip -j unsigned.apk classes.dex
echo "DEX added to APK"

echo "=== Step 6: Align APK ==="
rm -f "$BUILD_DIR/aligned.apk"
$ZIPALIGN -v -p 4 "$BUILD_DIR/unsigned.apk" "$BUILD_DIR/aligned.apk" 2>&1 | tail -3
echo "APK aligned"

echo "=== Step 7: Generate debug keystore ==="
if [ ! -f "/tmp/debug.keystore" ]; then
    keytool -genkey -v \
        -keystore /tmp/debug.keystore \
        -storepass android \
        -alias androiddebugkey \
        -keypass android \
        -keyalg RSA \
        -keysize 2048 \
        -validity 10000 \
        -dname "CN=Android Debug, O=Android, C=US" \
        2>&1 | tail -3
fi

echo "=== Step 8: Sign APK ==="
$APKSIGNER sign \
    --ks /tmp/debug.keystore \
    --ks-pass pass:android \
    --ks-key-alias androiddebugkey \
    --key-pass pass:android \
    --out "$OUT_APK" \
    "$BUILD_DIR/aligned.apk" \
    2>&1
echo "=== BUILD SUCCESSFUL ==="
ls -lh "$OUT_APK"

