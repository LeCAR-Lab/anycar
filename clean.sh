
cd $CAR_PATH

find . -type f -name "*.c" -exec rm -f {} \;
find . -type f -name "*.so" -exec rm -f {} \;
find . -type f -name "*.o" -exec rm -f {} \;