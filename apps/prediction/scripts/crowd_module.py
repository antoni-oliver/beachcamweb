import gc
import os
import json
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

from predictions.classes.BayesianPredictor import BayesianPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # apps/prediction/

try:
    from django.conf import settings as _dj_settings
    PREDICTION_CACHE_DIR = Path(_dj_settings.PREDICTION_DIR) / 'cache' / 'predictions'
except Exception:
    PREDICTION_CACHE_DIR = PROJECT_ROOT / 'cache' / 'predictions'

PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ABSTRACT BASE CLASS
# ============================================================

class BasePredictor(ABC):
    """Base class for crowd counting models."""

    name: str = "base"

    @abstractmethod
    def load(self):
        """Load model weights."""
        pass

    @abstractmethod
    def predict(self, image_data) -> float:
        """Predict crowd count from image."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """Get model info."""
        pass

    def predict_with_density(self, image_data):
        """Override in subclass if density map is supported."""
        raise NotImplementedError("Density map not supported for this model")


# ============================================================
# BAYESIAN VGG19 IMPLEMENTATION (wraps BayesianPredictor)
# ============================================================

class BayesianVGG19Predictor(BasePredictor):
    """Bayesian Crowd Counting with VGG19 backbone. Delegates to BayesianPredictor."""

    name = "bayesian_vgg19"

    def __init__(self, force_cpu=False):
        self._predictor = BayesianPredictor()
        self.force_cpu = force_cpu
        if force_cpu:
            self._predictor.device = "cpu"

    def _resolve_path(self, image_data) -> str:
        """Return a file path string, writing bytes to a temp file if needed."""
        if isinstance(image_data, (str, Path)):
            return str(image_data), None
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(image_data)
        tmp.close()
        return tmp.name, tmp.name  # (path, tmp_to_delete)

    def load(self):
        if self._predictor.model is not None:
            return
        self._predictor.prepareModel()
        print(f"[{self.name}] Model loaded on {self._predictor.device}")

    def predict(self, image_data) -> float:
        self.load()
        path, tmp = self._resolve_path(image_data)
        try:
            result = self._predictor.predict(path)
            return float(result.crowd_count)
        finally:
            if tmp:
                os.unlink(tmp)
            gc.collect()

    def predict_with_density(self, image_data):
        self.load()
        import torch
        path, tmp = self._resolve_path(image_data)
        try:
            inputs = self._predictor.processImage(path)
            with torch.no_grad():
                outputs = self._predictor.model(inputs)
                count = torch.sum(outputs).item()
                density = outputs.squeeze().cpu().numpy()
            return count, density
        finally:
            if tmp:
                os.unlink(tmp)
            gc.collect()

    def predict_full(self, image_data, mask_paths=None):
        self.load()
        import torch
        path, tmp = self._resolve_path(image_data)
        try:
            inputs = self._predictor.processImage(path)
            with torch.no_grad():
                outputs = self._predictor.model(inputs)
                count = torch.sum(outputs).item()
                density = outputs.squeeze().cpu().numpy()
            if mask_paths:
                density = self._predictor.applyMasks(mask_paths, density)
            overlay = self._predictor.mergeDensityMapWithImage(path, density)
            return count, density, overlay
        finally:
            if tmp:
                os.unlink(tmp)
            gc.collect()

    def apply_masks(self, mask_paths, density_map):
        return self._predictor.applyMasks(mask_paths, density_map)

    def merge_density_map(self, image_data, density_map):
        path, tmp = self._resolve_path(image_data)
        try:
            return self._predictor.mergeDensityMapWithImage(path, density_map)
        finally:
            if tmp:
                os.unlink(tmp)

    def predict_batch(self, image_paths, mask_paths=None):
        self.load()
        results = []
        for path in image_paths:
            try:
                result = self._predictor.predict(str(path), mask_paths or [])
                results.append(float(result.crowd_count))
            except Exception:
                results.append(None)
        return results

    def is_loaded(self) -> bool:
        return self._predictor.model is not None

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "loaded": self.is_loaded(),
            "device": str(self._predictor.device),
            "force_cpu": self.force_cpu,
            "weights_path": str(self._predictor.weigth_path),
        }


# ============================================================
# MODEL REGISTRY
# ============================================================

class PredictorRegistry:
    """Registry for managing multiple predictor models."""

    def __init__(self):
        self._predictors: dict[str, BasePredictor] = {}
        self._active: str = None

    def register(self, predictor: BasePredictor):
        """Register a predictor."""
        self._predictors[predictor.name] = predictor
        if self._active is None:
            self._active = predictor.name

    def set_active(self, name: str):
        """Set active predictor by name."""
        if name not in self._predictors:
            raise ValueError(f"Predictor '{name}' not registered. Available: {list(self._predictors.keys())}")
        self._active = name

    def get_active(self) -> BasePredictor:
        """Get active predictor."""
        if self._active is None:
            raise ValueError("No predictor registered")
        return self._predictors[self._active]

    def get(self, name: str) -> BasePredictor:
        """Get predictor by name."""
        if name not in self._predictors:
            raise ValueError(f"Predictor '{name}' not registered")
        return self._predictors[name]

    def list_predictors(self) -> dict:
        """List all registered predictors."""
        return {
            "active": self._active,
            "available": list(self._predictors.keys()),
            "predictors": {name: p.get_info() for name, p in self._predictors.items()}
        }

    def load_active(self):
        """Load the active predictor."""
        self.get_active().load()


# Global registry
_registry = PredictorRegistry()

# Register default predictor
# Use FORCE_CPU_BATCH=1 environment variable for batch processing to avoid MPS memory issues
_force_cpu = os.environ.get('FORCE_CPU_BATCH', '').lower() in ('1', 'true', 'yes')
_registry.register(BayesianVGG19Predictor(force_cpu=_force_cpu))

if _force_cpu:
    print("[crowd_module] FORCE_CPU_BATCH enabled - using CPU for predictions")


# ============================================================
# PUBLIC API (backwards compatible)
# ============================================================

def predict_count(image_data) -> float:
    """Predict crowd count using active model."""
    return round(_registry.get_active().predict(image_data), 4)


def predict_count_with_density(image_data):
    """Predict crowd count and density map using active model."""
    return _registry.get_active().predict_with_density(image_data)


def predict_full(image_data, mask_paths=None):
    """Predict crowd count, density map, and overlay PNG bytes."""
    predictor = _registry.get_active()
    if hasattr(predictor, 'predict_full'):
        return predictor.predict_full(image_data, mask_paths)
    count, density = predictor.predict_with_density(image_data)
    return count, density, None


def apply_masks(mask_paths, density_map):
    """Apply dark-area masks to a density map."""
    predictor = _registry.get_active()
    if hasattr(predictor, 'apply_masks'):
        return predictor.apply_masks(mask_paths, density_map)
    raise NotImplementedError("Active predictor does not support apply_masks")


def merge_density_map(image_data, density_map):
    """Overlay density map on image, returns PNG bytes."""
    predictor = _registry.get_active()
    if hasattr(predictor, 'merge_density_map'):
        return predictor.merge_density_map(image_data, density_map)
    raise NotImplementedError("Active predictor does not support merge_density_map")


def load_model():
    """Load the active model."""
    _registry.load_active()


def get_model_info() -> dict:
    """Get info about all registered models."""
    return _registry.list_predictors()


def register_predictor(predictor: BasePredictor):
    """Register a new predictor."""
    _registry.register(predictor)


def set_active_predictor(name: str):
    """Set active predictor by name."""
    _registry.set_active(name)


def get_predictor(name: str) -> BasePredictor:
    """Get predictor by name."""
    return _registry.get(name)


# ============================================================
# CACHE FUNCTIONS
# ============================================================

def _get_prediction_cache_key(filename, lat, lon):
    key = f"{filename}_{lat:.4f}_{lon:.4f}"
    return hashlib.md5(key.encode()).hexdigest()


def _sanitize_name(name):
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


def _get_cache_dir(model=None, name=None, subpath=None):
    """Get cache directory: cache/predictions/{model}/{name}/{subpath}/"""
    cache_dir = PREDICTION_CACHE_DIR
    if model:
        cache_dir = cache_dir / _sanitize_name(model)
    if name:
        cache_dir = cache_dir / _sanitize_name(name)
    if subpath:
        cache_dir = cache_dir / subpath
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_prediction_from_cache(filename, lat, lon, model=None, name=None, subpath=None):
    cache_key = _get_prediction_cache_key(filename, lat, lon)
    cache_dir = _get_cache_dir(model, name, subpath)
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


def save_prediction_to_cache(filename, lat, lon, result, model=None, name=None, subpath=None):
    cache_key = _get_prediction_cache_key(filename, lat, lon)
    cache_dir = _get_cache_dir(model, name, subpath)
    cache_file = cache_dir / f"{cache_key}.json"
    with open(cache_file, 'w') as f:
        json.dump(result, f)


def clear_prediction_cache(model=None, name=None):
    """Clear cache. Can filter by model and/or name."""
    count = 0
    if model or name:
        cache_dir = _get_cache_dir(model, name)
        for f in cache_dir.rglob("*.json"):
            f.unlink()
            count += 1
    else:
        for f in PREDICTION_CACHE_DIR.rglob("*.json"):
            f.unlink()
            count += 1
    return count


def list_cache_names():
    """List cache structure: {model: {name: count}}"""
    result = {}

    for model_dir in PREDICTION_CACHE_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        model_data = {}

        direct_count = len(list(model_dir.glob("*.json")))
        if direct_count > 0:
            model_data["_root"] = direct_count

        for name_dir in model_dir.iterdir():
            if name_dir.is_dir():
                total = len(list(name_dir.rglob("*.json")))
                if total > 0:
                    subdirs = {}
                    for sub in name_dir.rglob("*"):
                        if sub.is_dir():
                            sub_count = len(list(sub.glob("*.json")))
                            if sub_count > 0:
                                rel = str(sub.relative_to(name_dir))
                                subdirs[rel] = sub_count

                    if subdirs:
                        model_data[name_dir.name] = {"_total": total, "subdirs": subdirs}
                    else:
                        model_data[name_dir.name] = total

        if model_data:
            result[model_name] = model_data

    return result


def parse_datetime_from_filename(filename):
    try:
        stem = Path(filename).stem
        ts = int(stem)
        return datetime.fromtimestamp(ts)
    except:
        return None


# ============================================================
# INSPECTION UTILITIES (for analysis, doesn't affect API)
# ============================================================

def get_cache_stats(model=None):
    """Get statistics for all cameras in cache."""
    cache_dir = PREDICTION_CACHE_DIR
    if model:
        cache_dir = cache_dir / _sanitize_name(model)

    if not cache_dir.exists():
        return {}

    stats = {}
    for beach_dir in cache_dir.iterdir():
        if not beach_dir.is_dir():
            continue

        counts = []
        sample_paths = []

        for jf in beach_dir.rglob('*.json'):
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                if 'count' in data:
                    counts.append(data['count'])
                if len(sample_paths) < 5:
                    img_path = data.get('image_path')
                    if img_path and Path(img_path).exists():
                        sample_paths.append(img_path)
            except:
                pass

        if counts:
            stats[beach_dir.name] = {
                'records': len(counts),
                'max_count': max(counts),
                'min_count': min(counts),
                'mean_count': sum(counts) / len(counts),
                'sample_images': sample_paths
            }

    return stats


def find_low_count_cameras(model=None, threshold=60):
    """Find cameras with max count below threshold."""
    stats = get_cache_stats(model)
    return {
        name: data for name, data in stats.items()
        if data['max_count'] < threshold
    }


def get_sample_images_for_camera(camera_folder, model=None, n=3, images_base_dir=None):
    """Get sample image paths for a specific camera."""
    cache_dir = PREDICTION_CACHE_DIR
    if model:
        cache_dir = cache_dir / _sanitize_name(model)

    beach_path = cache_dir / camera_folder
    if not beach_path.exists():
        return []

    samples = []
    for jf in sorted(beach_path.rglob('*.json'))[:n * 5]:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)

            img_path = None
            filename = data.get('filename', '')

            # Try stored path first
            if 'image_path' in data and Path(data['image_path']).exists():
                img_path = data['image_path']

            # Try with images_base_dir
            elif images_base_dir and filename:
                base = Path(images_base_dir)

                # Try different folder patterns:
                # 1. Exact folder name: BeachCamDataset/camera_folder/file.jpg
                # 2. Nested with underscore: BeachCamDataset/Parent/child/file.jpg (from Parent_child)
                # 3. Check beach_folder from data

                possible_paths = [
                    base / camera_folder / filename,
                    base / camera_folder.replace('_', '/') / filename,
                ]

                # If data has beach_folder, try that too
                if 'beach_folder' in data:
                    bf = data['beach_folder']
                    possible_paths.extend([
                        base / bf / filename,
                        base / bf.replace('_', '/') / filename,
                    ])

                # Try splitting on first underscore for nested folders
                if '_' in camera_folder:
                    parts = camera_folder.split('_', 1)
                    possible_paths.append(base / parts[0] / parts[1] / filename)
                    # Also try deeper nesting
                    possible_paths.append(base / parts[0] / parts[1].replace('_', '/') / filename)

                for p in possible_paths:
                    if p.exists():
                        img_path = str(p)
                        break

            if img_path:
                samples.append({
                    'path': img_path,
                    'count': data.get('count', 0),
                    'filename': filename,
                    'datetime': data.get('datetime', '')
                })

            if len(samples) >= n:
                break
        except:
            pass

    return samples


def debug_image_paths(camera_folder, model=None, images_base_dir=None):
    """Debug function to show what paths are being tried."""
    cache_dir = PREDICTION_CACHE_DIR
    if model:
        cache_dir = cache_dir / _sanitize_name(model)

    beach_path = cache_dir / camera_folder
    if not beach_path.exists():
        print(f"Cache folder not found: {beach_path}")
        return

    # Get first JSON file
    json_files = list(beach_path.rglob('*.json'))[:1]
    if not json_files:
        print(f"No JSON files in {beach_path}")
        return

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    print(f"Cache folder: {camera_folder}")
    print(f"JSON data keys: {list(data.keys())}")
    print(f"  filename: {data.get('filename', 'N/A')}")
    print(f"  beach_folder: {data.get('beach_folder', 'N/A')}")
    print(f"  image_path: {data.get('image_path', 'N/A')}")

    if images_base_dir:
        base = Path(images_base_dir)
        filename = data.get('filename', '')
        bf = data.get('beach_folder', '')

        print(f"\nTrying paths with base: {base}")
        paths_to_try = [
            base / camera_folder / filename,
            base / camera_folder.replace('_', '/') / filename,
            base / bf / filename if bf else None,
            base / bf.replace('_', '/') / filename if bf else None,
        ]

        if '_' in camera_folder:
            parts = camera_folder.split('_', 1)
            paths_to_try.append(base / parts[0] / parts[1] / filename)

        for p in paths_to_try:
            if p:
                exists = "✓ EXISTS" if p.exists() else "✗ not found"
                print(f"  {p} -> {exists}")


def analyze_camera_with_density(camera_folder, model=None, n_samples=2, images_base_dir=None, save_dir=None):
    """
    Analyze a camera by running Bayesian model with density maps.
    Returns analysis results and optionally saves visualization.
    """
    samples = get_sample_images_for_camera(camera_folder, model, n_samples, images_base_dir)

    if not samples:
        return {'camera': camera_folder, 'error': 'No sample images found'}

    results = []
    for sample in samples:
        try:
            count, density = predict_count_with_density(sample['path'])
            results.append({
                'path': sample['path'],
                'filename': sample['filename'],
                'datetime': sample['datetime'],
                'cached_count': sample['count'],
                'recomputed_count': round(count, 2),
                'density_shape': density.shape,
                'density_max': float(density.max()),
                'density_sum': float(density.sum()),
                'density': density
            })
        except Exception as e:
            results.append({
                'path': sample['path'],
                'error': str(e)
            })

    return {
        'camera': camera_folder,
        'samples_analyzed': len(results),
        'results': results
    }


def inspect_low_count_cameras(model=None, threshold=60, n_samples=2, images_base_dir=None, show_plots=True):
    """
    Full inspection of low-count cameras with density visualization.

    Args:
        model: Model name in cache (e.g., 'bayesian_vgg19')
        threshold: Max count threshold to filter cameras
        n_samples: Number of samples per camera
        images_base_dir: Base directory for original images
        show_plots: Whether to display matplotlib plots

    Returns:
        Dict with camera analysis results
    """
    low_count = find_low_count_cameras(model, threshold)

    if not low_count:
        print(f"No cameras found with max < {threshold}")
        return {}

    print(f"Found {len(low_count)} cameras with max < {threshold}:")
    for name, data in sorted(low_count.items(), key=lambda x: x[1]['max_count']):
        print(f"  {name:30s} | max: {data['max_count']:6.1f} | mean: {data['mean_count']:6.1f} | n={data['records']}")

    if not show_plots:
        return {'cameras': low_count, 'analyses': {}}

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from PIL import Image
    except ImportError:
        print("matplotlib/PIL not available for plotting")
        return {'cameras': low_count, 'analyses': {}}

    analyses = {}

    for camera_name in low_count.keys():
        print(f"\nAnalyzing: {camera_name}")
        analysis = analyze_camera_with_density(camera_name, model, n_samples, images_base_dir)
        analyses[camera_name] = analysis

        if 'error' in analysis:
            print(f"  Error: {analysis['error']}")
            continue

        valid_results = [r for r in analysis['results'] if 'density' in r]
        if not valid_results:
            print(f"  No valid density maps generated")
            continue

        fig, axes = plt.subplots(len(valid_results), 2, figsize=(12, 5 * len(valid_results)))
        if len(valid_results) == 1:
            axes = [axes]

        for idx, result in enumerate(valid_results):
            try:
                img = Image.open(result['path'])

                axes[idx][0].imshow(img)
                axes[idx][0].set_title(f"Original - Count: {result['recomputed_count']:.1f}\n{result.get('datetime', '')}", fontsize=10)
                axes[idx][0].axis('off')

                im = axes[idx][1].imshow(result['density'], cmap='jet', norm=mcolors.PowerNorm(gamma=0.5))
                axes[idx][1].set_title(f"Density Map (max: {result['density_max']:.2f})", fontsize=10)
                axes[idx][1].axis('off')
                plt.colorbar(im, ax=axes[idx][1], fraction=0.046)

                img.close()
            except Exception as e:
                print(f"  Plot error: {e}")

        plt.suptitle(f"{camera_name} - Max in dataset: {low_count[camera_name]['max_count']:.0f}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 60)
    print("DECISION GUIDE:")
    print("- Density on SAND/WATER with people shapes → KEEP")
    print("- Density on MOUNTAINS/TREES/BUILDINGS → EXCLUDE")
    print("- Random scattered dots everywhere → EXCLUDE")
    print("=" * 60)

    return {'cameras': low_count, 'analyses': analyses}

def predict_count_batch(image_paths, mask_paths=None) -> list:
    """Batch predict crowd counts using active model."""
    predictor = _registry.get_active()
    if hasattr(predictor, 'predict_batch'):
        return predictor.predict_batch(image_paths, mask_paths)
    return [predictor.predict(p) for p in image_paths]

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    load_model()
    print(json.dumps(get_model_info(), indent=2))

    test_image = PROJECT_ROOT / "sample_images/1657879200.jpg"
    if test_image.exists():
        count = predict_count(test_image)
        print(f"Count: {count}")