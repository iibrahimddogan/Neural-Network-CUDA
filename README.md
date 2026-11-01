# Neural-Network-CUDA

A CNN framework written from scratch in C++/CUDA: custom kernels for convolution, pooling, linear layers, activations, softmax, and cross-entropy; manual forward/backward propagation and model serialization.

## Architecture

```
Conv2D(1→16, 3×3) → MaxPool(2×2)
→ Conv2D(16→32, 3×3) → MaxPool(2×2)
→ Flatten(800) → Linear(800→128, ReLU)
→ Linear(128→64, ReLU) → Linear(64→10)
```

## Results

| Dataset | Epochs | Batch | Test accuracy |
|---|---|---|---|
| MNIST (`data/number_data/`) | 20 | 128 | **98.72%** |
| Fashion-MNIST (`data/`) | 20 | 128 | **88.55%** |

Training: 20 epochs, batch size 128, learning rate 0.05.

The default loader reads `data/`, which contains Fashion-MNIST in this repository. MNIST digits are in `data/number_data/`; the MNIST result above was produced by using those files as `data/`.

## Design notes

- 19 custom CUDA kernels for conv, pooling, matmul, ReLU/Sigmoid, softmax/cross-entropy, and gradients.
- Softmax/cross-entropy uses a shared-memory reduction.
- Matmul and convolution are naive baselines; tiled versions are next.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

If your host compiler is newer than the CUDA toolkit supports:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
```

## Run

From the repository root:

```bash
# Linux/macOS (single-config generators)
./build/prog        # train 20 epochs and save data/mnist_cnn_oop_model.bin
./build/prog test   # evaluate on the 10k test images

# Windows / Visual Studio (multi-config generators)
.\build\Release\prog.exe
.\build\Release\prog.exe test
```

## Roadmap

- [ ] Tiled matmul/conv + benchmark
- [ ] `--dataset mnist|fashion` flag
- [ ] Conv bias + He initialization + shuffle
- [ ] Per-epoch train accuracy

## License

MIT
