##! /bin/bash
echo "Start build database.."
echo "1. create trainging set..."
python3 ./trainingSet/create_training_set.py

echo "2. create identification photos..."
python3 ./IDPhotos/create_id_photos.py

echo "3. create identification labels..."
python3 ./IDLabels/create_id_labels.py


echo "Build database done."
