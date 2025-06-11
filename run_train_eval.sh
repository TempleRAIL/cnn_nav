#!/bin/sh
#
# file: run.sh
#
# This is a simple driver script that runs training and then decoding
# on the training set, the dev test set and the eval set.
#
# To run this script, execute the following line:
#
#  run.sh train.dat test.dat eval.dat
#
# The first argument ($1) is the training data. The last two arguments,
# test data ($2) and evaluation data ($3) are optional.
#
# An example of how to run this is as follows:
#
# nedc_000_[1]: echo $PWD
# /data/isip/exp/tuh_dpath/exp_0074/v1.0
# nedc_000_[1]: ./run.sh data/train_set.txt data/dev_set.txt data/eval_set.txt
#
# This script will take you through the sequence of steps required to
# train a simple MLP network and evaluate it on some data.
#
# The script will then take the trained models and do an evaluation
# on the data in "test.dat". It will output the results to output/results.txt.
#
# If an eval set is specified, it will do the same for the eval set.
#

# decode the number of command line arguments
#
NARGS=$#

if (test "$NARGS" -eq "0") then
    echo "usage: run.sh train.dat [test.dat] [eval.dat]"
    exit 1
fi

# define a base directory for the experiment
#
DL_EXP=`pwd`;
DL_SCRIPTS="$DL_EXP/scripts";
DL_OUT="$DL_EXP/output";
DL_LABELS="$DL_EXP/labels";

# define the number of feats environment variable
#
export DL_NUM_FEATS=5 #26

# define the output directories for training/decoding/scoring
#
#DL_TRAIN_ODIR="$DL_OUT/00_train";
DL_TRAIN_ODIR="$DL_EXP/model";
DL_MDL_PATH="$DL_TRAIN_ODIR/model.pth";

DL_DECODE_ODIR="$DL_OUT/01_hyp";
DL_HYP_TRAIN="$DL_DECODE_ODIR/train_set.hyp";
DL_HYP_DEV="$DL_DECODE_ODIR/dev_set.hyp";
DL_HYP_EVAL="$DL_DECODE_ODIR/eval_set.hyp";

# create the output directory
#
#rm -fr $DL_OUT
#mkdir -p $DL_OUT

# execute training: training must always be run
#
echo "... starting training on $1 ..."
$DL_SCRIPTS/train.py $DL_MDL_PATH $1 $2 | tee $DL_OUT/00_train.log | \
      grep "reading\|Step\|Average\|Warning\|Error" 
echo "... finished training on $1 ..."

#
