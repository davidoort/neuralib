from matplotlib import pyplot as plt
import numpy as np

from neuralib.architectures import Model

def _angular_error(X, y):
    """Calculate angular error (via cosine similarity)."""

    def pitchyaw_to_vector(pitchyaws):
        """Convert given pitch and yaw angles to unit gaze vectors."""
        n = pitchyaws.shape[0]
        sin = np.sin(pitchyaws)
        cos = np.cos(pitchyaws)
        out = np.empty((n, 3))
        out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
        out[:, 1] = sin[:, 0]
        out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
        return out

    a = pitchyaw_to_vector(y) 
    b = pitchyaw_to_vector(X)

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * (180.0 / np.pi)


def predict_and_calculate_mean_error(model: Model, x, y):
    """Calculate mean error of neural network predictions on given data."""
    n, _, _ = x.shape
    predictions = model.predict(x.reshape(n, -1)).reshape(-1, 2)
    labels = y.reshape(-1, 2)
    errors = _angular_error(predictions, labels)
    return np.mean(errors)


def predict_and_visualize(model: Model, x, y):
    """Visualize errors of neural network on given data."""
    nr, nc = 1, 12
    n = nr * nc
    fig = plt.figure(figsize=(12, 2.))
    predictions = model.predict(x[:n, :].reshape(n, -1))
    for i, (image, label, prediction) in enumerate(zip(x[:n], y[:n], predictions)):
        plt.subplot(nr, nc, i + 1)
        plt.imshow(image, cmap='gray')
        error = _angular_error(prediction.reshape(1, 2), label.reshape(1, 2))
        plt.title('%.1f' % error, color='g' if error < 7.0 else 'r')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout(pad=0.0)
    plt.show()