## Install evaluation tools

before you calculate the penetration and intersection between two objects,
you need to process the object models so that they are **watertight** and then **voxelized**.

- **watertight**: the object model is closed, and there is no hole in the model. (Using tool ManofoldPlus)
- **voxelized**: the object model is represented by a voxel grid. (Using tool binvox).

### Watertight

Clone the repo [ManofoldPlus](https://github.com/hjwdzh/ManifoldPlus) into `./thridparty` 
and follow its instructions to build the tool.  
Then, you can use the following command to process the object model:

```bash
./thridparty/ManifoldPlus/build/manifold --input $OBJ_PATH --output $WT_OBJ_PATH --depth 8
```

This script only support `.obj` file, if you have `.ply`, please convert it to `.obj` first. The output file will be at the `$WT_OBJ_PATH` you specified.

### Voxelized

Download the pre-build [**binvox**](https://www.patrickmin.com/binvox/) executable ( Download -> Linux 64 bit executable) into `./thridparty`.  
Then, you can use the following command to process the (watertight) object model:

```bash
./thridparty/binvox -d 128 $WT_OBJ_PATH
```

The output file will be stored as `.binvox` in the same directory as the input file.

### Libmesh

To calculate the penetration and intersection, you need to install the `libmesh` package. 
```bash
conda activate oishape_bm
cd thirdparty/libmesh
python setup.py build_ext --inplace
``` 
The `libmesh` was originally developed by [occupancy_networks](https://github.com/autonomousvision/occupancy_networks/tree/master/im2mesh/utils/libmesh) (CVPR'19). We simply modified it to support the new version of numpy (>1.20)