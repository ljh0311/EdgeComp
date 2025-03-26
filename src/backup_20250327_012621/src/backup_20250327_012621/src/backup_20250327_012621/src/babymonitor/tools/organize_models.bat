@echo off
echo Organizing models directory...

:: Create directories
mkdir "models\emotion\speechbrain" 2>nul
mkdir "models\emotion\cry_detection" 2>nul
mkdir "models\haarcascade" 2>nul
mkdir "models\mobilenet_ssd" 2>nul

:: Move emotion models
move "models\hubert-base-ls960_emotion.pt" "models\emotion\speechbrain\" 2>nul
move "models\emotion_model.pt" "models\emotion\speechbrain\" 2>nul
move "models\best_emotion_model.pt" "models\emotion\speechbrain\" 2>nul
move "models\cry_detection_model.pth" "models\emotion\cry_detection\" 2>nul

:: Move Haar cascade models
move "models\haarcascade_*.xml" "models\haarcascade\" 2>nul

:: Move MobileNet SSD models
move "models\MobileNetSSD_deploy.caffemodel" "models\mobilenet_ssd\" 2>nul
move "models\person_labels.txt" "models\mobilenet_ssd\" 2>nul
move "models\person_detection_model.tflite" "models\mobilenet_ssd\" 2>nul

echo Models organized successfully!
pause 