#!/bin/bash
set -x

if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

if [ -x "$SMART_ROOT/submodules" ];then
    submodules_root=$SMART_ROOT
else    
    submodules_root=$PWD
fi

if [[ "$submodules_root" =~ "submodules/mmtrack" ]]
then
    if [ ! -d "$submodules_root/../mmdet/mmdet" ]
    then
        cd ../..
        git submodule update --init submodules/mmdet
        cd -
    fi
    if [ ! -d "$submodules_root/../mmcls/mmcls" ]
    then
        cd ../..
        git submodule update --init submodules/mmcls
        cd -
    fi
else
    if [ ! -d "$submodules_root/submodules/mmdet/mmdet" ]
    then
        git submodule update --init submodules/mmdet
    fi
    if [ ! -d "$submodules_root/submodules/mmcls/mmcls" ]
    then
        git submodule update --init submodules/mmcls
    fi
fi

# 0. placeholder
if [ -d "$submodules_root/submodules/mmtrack/algolib/configs" ]
then
    rm -rf $submodules_root/submodules/mmtrack/algolib/configs
    ln -s $submodules_root/submodules/mmtrack/configs $submodules_root/submodules/mmtrack/algolib/
else
    ln -s $submodules_root/submodules/mmtrack/configs $submodules_root/submodules/mmtrack/algolib/
fi
 
# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmtrack/$3
export PYTORCH_VERSION=1.4
 
# 2. set time
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env
path=$PWD
if [[ "$path" =~ "submodules" ]]
then
    pyroot=$submodules_root/mmtrack
else
    pyroot=$submodules_root/submodules/mmtrack
fi
echo $pyroot
export PYTHONPATH=$pyroot:$PYTHONPATH
export FRAME_NAME=mmtrack    #customize for each frame
export MODEL_NAME=$3

# mmdet_path and mmcls_path
SHELL_PATH=$(dirname $0)
export PYTHONPATH=$SHELL_PATH/../../../mmdet:$PYTHONPATH
export PYTHONPATH=$SHELL_PATH/../../../mmcls:$PYTHONPATH

# init_path
export PYTHONPATH=$SMART_ROOT/common/sites/:$PYTHONPATH 
 
# 4. build necessary parameter
partition=$1 
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
 
# 5. model choice
export PARROTS_DEFAULT_LOGGER=FALSE

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

port=`expr $RANDOM % 10000 + 20000`

file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --cfg-options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
