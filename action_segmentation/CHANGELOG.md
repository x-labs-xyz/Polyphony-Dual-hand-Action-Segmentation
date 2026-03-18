# Changelog

All notable changes to the Dual-Hand Action Segmentation project.

## [1.1.0] - 2024-11-07

### Added
- ✨ **Single-Stream Mode**: Support for training on single-stream datasets (e.g., Breakfast, 50Salads)
  - `--one_stream` flag to enable single-stream mode
  - Automatic label perturbation by shifting action boundaries
  - Configurable perturbation parameters: `--max_shift_frames`, `--max_perturbations`, `--min_segment_len`
  - Creates synthetic dual-stream data for improved training
- 📖 Comprehensive documentation for single-stream mode in README.md and QUICK_START.md
- 🎯 Console feedback showing perturbation parameters when single-stream mode is active

### Technical Details

**Perturbation Algorithm:**
- Randomly selects up to N action boundaries per video
- Shifts each boundary by ±K frames
- Preserves minimum segment lengths
- Maintains temporal ordering and sequence continuity

**Use Cases:**
- Datasets with only one action stream
- Data augmentation for small datasets
- Creating robust models through synthetic variations

## [1.0.0] - 2024-11-07

### Initial Release

#### Features
- ✨ Dual-hand action segmentation with shared encoder
- 🔄 Diffusion-based decoder with DDIM sampling
- ⚖️ Adaptive loss weighting for balanced training
- 📊 Comprehensive evaluation metrics (Acc, Edit, F1@IoU)
- 🛠️ Temporal augmentation and post-processing
- 📈 TensorBoard integration
- 💾 Checkpoint resuming
- 🎯 Hand-specific class weighting

#### Components

**Core Modules:**
- `model.py` - DualHandASDiffusionModel architecture
- `dataset.py` - Data loading and augmentation
- `main.py` - Training pipeline with AdaptiveLossWeightManager
- `utils.py` - Evaluation metrics and utilities
- `config.py` - Configuration management

**Documentation:**
- `README.md` - Comprehensive documentation
- `QUICK_START.md` - 5-step quick start guide
- `CHANGELOG.md` - Version history (this file)

**Supporting Files:**
- `requirements.txt` - Python dependencies
- `train.sh` - Training script
- `.gitignore` - Git ignore patterns

#### Architecture

- **Encoder**: Shared temporal convolutional network
- **Feature Fusion**: Cross-hand context integration
- **Decoder**: Hand-specific diffusion-based refinement
- **Loss Manager**: Performance-based adaptive weighting

#### Evaluation Metrics

- Frame-wise accuracy
- Edit score (normalized Levenshtein distance)
- F1 scores @10%, 25%, 50% IoU

#### Post-Processing

- Median filter
- Mode filter
- Short segment removal (purge)

#### Configuration

- JSON-based configuration
- Python config generator
- Extensive hyperparameter options

### Technical Details

**Diffusion Process:**
- Cosine beta schedule
- DDIM sampling (25 steps default)
- Multiple conditioning strategies
- SNR scale: 2.0

**Training:**
- Adam optimizer
- Gradient accumulation support
- Class weighting for imbalanced data
- Instance normalization option
- Soft label smoothing option

**Data Augmentation:**
- Temporal augmentation (multiple offsets)
- Spatial augmentation (if available)
- Boundary smoothing (optional)

### Known Limitations

- Requires pre-extracted visual features
- Single GPU training (multi-GPU support planned)
- Placeholders for EncoderModel and DecoderModel (require custom implementation)

### Future Work

- [ ] Multi-GPU training support
- [ ] End-to-end video input (feature extraction integrated)
- [ ] Pre-trained encoder/decoder models
- [ ] Real-time inference optimization
- [ ] Cross-dataset evaluation scripts
- [ ] Data preprocessing utilities

---

## Version Guidelines

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes (backward-compatible)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Release Notes Format

Each release includes:
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security updates

---

**Last Updated**: 2024-11-07

