#!/bin/bash
#SBATCH -J DopplerSegmentation
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /homedtic/imunoz/DADES/DADES/Echo/Logs/%J.out
#SBATCH -e /homedtic/imunoz/DADES/DADES/Echo/Logs/%J.err
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

module load GCCcore/9.3.0 libffi/3.3-GCCcore-9.3.0 bokeh/2.0.2-foss-2020a-Python-3.8.2 PCRE/8.44-GCCcore-9.3.0 \
            zlib/1.2.11-GCCcore-9.3.0 OpenSSL/1.1.1e-GCCcore-9.3.0 dask/2.18.1-foss-2020a-Python-3.8.2 \
            GLib/2.64.1-GCCcore-9.3.0 binutils/2.34-GCCcore-9.3.0 Python/3.8.2-GCCcore-9.3.0 ATK/2.36.0-GCCcore-9.3.0 \
            GCC/9.3.0 Ninja/1.10.0-GCCcore-9.3.0 DBus/1.13.12-GCCcore-9.3.0 CUDAcore/11.0.2 UCX/1.8.0-GCCcore-9.3.0 \
            protobuf/3.13.0-GCCcore-9.3.0 at-spi2-core/2.36.0-GCCcore-9.3.0 CUDA/11.0.2-GCC-9.3.0 OpenMPI/4.0.3-GCC-9.3.0 \
            protobuf-python/3.13.0-fosscuda-2020a-Python-3.8.2 at-spi2-atk/2.34.2-GCCcore-9.3.0 gcccuda/2020a gompi/2020a \
            typing-extensions/3.7.4.3-GCCcore-9.3.0-Python-3.8.2 Gdk-Pixbuf/2.40.0-GCCcore-9.3.0 numactl/2.0.13-GCCcore-9.3.0 \
            FFTW/3.3.8-gompi-2020a MPFR/4.0.2-GCCcore-9.3.0 pixman/0.38.4-GCCcore-9.3.0 XZ/5.2.5-GCCcore-9.3.0 \
            ScaLAPACK/2.1.0-gompi-2020a x264/20191217-GCCcore-9.3.0 cairo/1.16.0-GCCcore-9.3.0 libxml2/2.9.10-GCCcore-9.3.0 \
            foss/2020a LAME/3.100-GCCcore-9.3.0 ICU/66.1-GCCcore-9.3.0 libpciaccess/0.16-GCCcore-9.3.0 \
            libpng/1.6.37-GCCcore-9.3.0 x265/3.3-GCCcore-9.3.0 HarfBuzz/2.6.4-GCCcore-9.3.0 hwloc/2.2.0-GCCcore-9.3.0 \
            freetype/2.10.1-GCCcore-9.3.0 FriBidi/1.0.9-GCCcore-9.3.0 Pango/1.44.7-GCCcore-9.3.0 libevent/2.1.11-GCCcore-9.3.0 \
            expat/2.2.9-GCCcore-9.3.0 FFmpeg/4.2.2-GCCcore-9.3.0 gzip/1.10-GCCcore-9.3.0 Check/0.15.2-GCCcore-9.3.0 \
            util-linux/2.35-GCCcore-9.3.0 cuDNN/8.0.4.30-CUDA-11.0.2 lz4/1.9.2-GCCcore-9.3.0 GDRCopy/2.1-GCCcore-9.3.0-CUDA-11.0.2 \
            fontconfig/2.13.92-GCCcore-9.3.0 magma/2.5.4-fosscuda-2020a zstd/1.4.4-GCCcore-9.3.0 libfabric/1.11.0-GCCcore-9.3.0 \
            xorg-macros/1.19.2-GCCcore-9.3.0 NCCL/2.8.3-GCCcore-9.3.0-CUDA-11.0.2 libdrm/2.4.100-GCCcore-9.3.0 \
            PMIx/3.1.5-GCCcore-9.3.0 X11/20200222-GCCcore-9.3.0 libglvnd/1.2.0-GCCcore-9.3.0 OpenBLAS/0.3.9-GCC-9.3.0 \
            Tk/8.6.10-GCCcore-9.3.0 pybind11/2.4.3-GCCcore-9.3.0-Python-3.8.2 libunwind/1.3.1-GCCcore-9.3.0 gompic/2020a \
            Tkinter/3.8.2-GCCcore-9.3.0 SciPy-bundle/2020.03-foss-2020a-Python-3.8.2 LLVM/9.0.1-GCCcore-9.3.0 fosscuda/2020a \
            matplotlib/3.2.1-foss-2020a-Python-3.8.2 giflib/5.2.1-GCCcore-9.3.0 Mesa/20.0.2-GCCcore-9.3.0 bzip2/1.0.8-GCCcore-9.3.0 \
            NASM/2.14.02-GCCcore-9.3.0 libwebp/1.1.0-GCCcore-9.3.0 libepoxy/1.5.4-GCCcore-9.3.0 ncurses/6.2-GCCcore-9.3.0 \
            libjpeg-turbo/2.0.4-GCCcore-9.3.0 OpenEXR/2.4.1-GCCcore-9.3.0 GTK+/3.24.17-GCCcore-9.3.0 libreadline/8.0-GCCcore-9.3.0 \
            LibTIFF/4.1.0-GCCcore-9.3.0 JasPer/2.0.14-GCCcore-9.3.0 Tcl/8.6.10-GCCcore-9.3.0 Pillow/7.0.0-GCCcore-9.3.0-Python-3.8.2 \
            Java/11.0.2 SQLite/3.31.1-GCCcore-9.3.0 libyaml/0.2.2-GCCcore-9.3.0 ant/1.10.8-Java-11 GMP/6.2.0-GCCcore-9.3.0 \
            PyYAML/5.3-GCCcore-9.3.0 gettext/0.20.1-GCCcore-9.3.0 PyTorch/1.9.0-fosscuda-2020a-Python-3.8.2;
source ~/miniconda3/bin/activate;

cd ~/GitHub/DopplerSegmentation_iago;

python3 train_doppler_lr.py --config_file ./configurations/configuration_HPC_7.json 



