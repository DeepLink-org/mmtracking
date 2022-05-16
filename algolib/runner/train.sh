#!/bin/bash
set -x
set -o pipefail
set -e

# 0. check the most important SMART_ROOT
echo  "!!!!!SMART_ROOT is" $SMART_ROOT
if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

# 1. set env_path and build soft links for mm configs
if [[ $PWD =~ "mmtrack" ]]
then 
    pyroot=$PWD
    if [ ! -d "$PWD/../mmdet/mmdet" ]
    then
        cd ..
        git submodule update --init mmdet
        cd -
    fi
    if [ ! -d "$PWD/../mmcls/mmcls" ]
    then
        cd ..
        git submodule update --init mmcls
        cd -
    fi
else
    pyroot=$PWD/mmtrack
    if [ ! -d "$PWD/mmdet/mmdet" ]
    then
        git submodule update --init mmdet
    fi
    if [ ! -d "$PWD/mmcls/mmcls" ]
    then
        git submodule update --init mmcls
    fi
fi
echo $pyroot
if [ -d "$pyroot/algolib/configs" ]
then
    rm -rf $pyroot/algolib/configs
    ln -s $pyroot/configs $pyroot/algolib/
else
    ln -s $pyroot/configs $pyroot/algolib/
fi
 
# 2. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmtrack/$3
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env variables
export PYTORCH_VERSION=1.4
export PYTHONPATH=$pyroot:$PYTHONPATH
export FRAME_NAME=mmtrack    #customize for each frame
export MODEL_NAME=$3
export PARROTS_DEFAULT_LOGGER=FALSE

# 4. mmdetpath and init_path
export PYTHONPATH=$pyroot/../mmdet:$PYTHONPATH
export PYTHONPATH=$pyroot/../mmcls:$PYTHONPATH
export PYTHONPATH=${SMART_ROOT}:$PYTHONPATH
export PYTHONPATH=$SMART_ROOT/common/sites/:$PYTHONPATH

# 5. build necessary parameter
partition=$1 
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
EXTRA_ARGS=${array[@]:3}
EXTRA_ARGS=${EXTRA_ARGS//--resume/--resume-from}
SRUN_ARGS=${SRUN_ARGS:-""}
 
# 6. model list
case $MODEL_NAME in
    "dff_faster_rcnn_r101_dc5_1x_imagenetvid")
        FULL_MODEL="vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid"
        ;;
    "selsa_faster_rcnn_r101_dc5_1x_imagenetvid")
        FULL_MODEL="vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid"
        ;;
    "faster-rcnn_r50_fpn_4e_mot17-half")
        FULL_MODEL="det/faster-rcnn_r50_fpn_4e_mot17-half"
        ;;
    "siamese_rpn_r50_1x_lasot")
        FULL_MODEL="sot/siamese_rpn/siamese_rpn_r50_1x_lasot"
        ;;
    # "resnet50_b32x8_MOT17")
    #     FULL_MODEL="reid/resnet50_b32x8_MOT17"
    #     ;;
    # 注：该模型存在问题，详见：https://jira.sensetime.com/browse/PARROTSXQ-8232
    # "masktrack_rcnn_r50_fpn_12e_youtubevis2019")
    #     FULL_MODEL="vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019"
    #     ;;
    # 注：该模型存在问题，详见：https://jira.sensetime.com/browse/PARROTSXQ-8231
    # "bytetrack_yolox_x_crowdhuman_mot17-private-half")
    #     FULL_MODEL="mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half"
    #     ;;
    # 注：该模型存在问题，详见：https://jira.sensetime.com/browse/PARROTSXQ-8230
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

# 7. set port and choice model
port=`expr $RANDOM % 10000 + 20000`
file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

# 8. run model
srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --cfg-options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
