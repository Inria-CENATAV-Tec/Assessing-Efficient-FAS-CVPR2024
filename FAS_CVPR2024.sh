PHASE_1_TRAIN_PASSWORD=""
PHASE_1_DEV_PASSWORD=""
PHASE_2_TEST_PASSWORD=""

#Unzip face detection model
unzip models/retinaface-R50.zip -d models/retinaface-R50/

#Download and pre-process data
cd datasets/FAS-CVPR2024/

unzip -P $PHASE_1_TRAIN_PASSWORD UniAttackData/phase1/p1/train.zip -d UniAttackData/phase1/p1/
unzip -P $PHASE_1_TRAIN_PASSWORD UniAttackData/phase1/p2.1/train.zip -d UniAttackData/phase1/p2.1/
unzip -P $PHASE_1_TRAIN_PASSWORD UniAttackData/phase1/p2.2/train.zip -d UniAttackData/phase1/p2.2/
unzip -P $PHASE_1_DEV_PASSWORD UniAttackData/phase1/p1/dev.zip -d UniAttackData/phase1/p1/
unzip -P $PHASE_1_DEV_PASSWORD UniAttackData/phase1/p2.1/dev.zip -d UniAttackData/phase1/p2.1/
unzip -P $PHASE_1_DEV_PASSWORD UniAttackData/phase1/p2.2/dev.zip -d UniAttackData/phase1/p2.2/ 

unzip -P $PHASE_2_TEST_PASSWORD UniAttackData/phase2/p1/test.zip -d UniAttackData/phase2/p1/
unzip -P $PHASE_2_TEST_PASSWORD UniAttackData/phase2/p2.1/test.zip -d UniAttackData/phase2/p2.1/
unzip -P $PHASE_2_TEST_PASSWORD UniAttackData/phase2/p2.2/test.zip -d UniAttackData/phase2/p2.2/

#Face Detection
python detect_norm_crop.py

cp UniAttackData/phase1/p1/*.txt norm_crop/UniAttackData/phase1/p1/
cp UniAttackData/phase1/p2.1/*.txt norm_crop/UniAttackData/phase1/p2.1/
cp UniAttackData/phase1/p2.2/*.txt norm_crop/UniAttackData/phase1/p2.2/
cp UniAttackData/phase2/p1/*.txt norm_crop/UniAttackData/phase2/p1/
cp UniAttackData/phase2/p2.1/*.txt norm_crop/UniAttackData/phase2/p2.1/
cp UniAttackData/phase2/p2.2/*.txt norm_crop/UniAttackData/phase2/p2.2/
cp UniAttackData/phase2/p1/*.txt norm_crop/UniAttackData/phase1/p1/
cp UniAttackData/phase2/p2.1/*.txt norm_crop/UniAttackData/phase1/p2.1/
cp UniAttackData/phase2/p2.2/*.txt norm_crop/UniAttackData/phase1/p2.2/

cd ../../

#Training
python train.py --config configs/config_cvpr2024_p1.py
python train.py --config configs/config_cvpr2024_p2_1.py
python train.py --config configs/config_cvpr2024_p2_2.py

#Inference
python save_predictions_FASCVPR2024.py --config configs/config_cvpr2024_p1.py
