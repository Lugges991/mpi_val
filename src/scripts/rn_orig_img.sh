#!/bin/bash

for f in *.nii
do
    mv "$f" "${f%.nii}_orig.nii"
done
