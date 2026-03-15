#!/bin/bash
# Generate teraxlang-dev wheel from original wheel

DEV_SUFFIX="dev3"  # Change this to create different dev versions
ORIGINAL_WHL="thirdparty/triton/dist/teraxlang-3.5.1-cp312-cp312-manylinux_2_35_x86_64.whl"
OUTPUT_WHL="teraxlang-3.5.1.${DEV_SUFFIX}-cp312-cp312-manylinux_2_35_x86_64.whl"

# 1. Unpack original wheel to temp directory
rm -rf tmp_wheel && mkdir -p tmp_wheel
cd tmp_wheel
unzip -q ../$ORIGINAL_WHL

# 2. Remove static .a files (major space savings)
find . -name "*.a" -delete

# 3. Update version in METADATA and update RECORD references
# Use perl instead of sed for cross-platform compatibility
perl -i -pe "s/Version: 3.5.1/Version: 3.5.1.${DEV_SUFFIX}/" teraxlang-3.5.1.dist-info/METADATA

# Update RECORD file to reference new dist-info folder name
perl -i -pe "s/teraxlang-3\.5\.1\.dist-info/teraxlang-3.5.1.${DEV_SUFFIX}.dist-info/g" teraxlang-3.5.1.dist-info/RECORD

# Rename dist-info folder to match new version
mv teraxlang-3.5.1.dist-info teraxlang-3.5.1.${DEV_SUFFIX}.dist-info

# 4. Repack wheel
cd ..
rm -f $OUTPUT_WHL
cd tmp_wheel
zip -r ../$OUTPUT_WHL .

# 6. Cleanup
cd ..
rm -rf tmp_wheel

# 7. Show result
ls -lh $OUTPUT_WHL
