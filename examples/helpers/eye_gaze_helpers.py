from xmlrpc.client import Boolean
from matplotlib import pyplot as plt
import numpy as np

from neuralib.metrics import ScalarMetric

class AngularError(ScalarMetric):
    """Calculate angular error (via cosine similarity)."""

    def __init__(self, every_n_epochs: int = 1, img_visualize: Boolean = False) -> None:
        super().__init__(every_n_epochs)
        self.img_visualize = img_visualize
        
    def calculate_from_predictions(self, y_pred: np.array, y: np.array):
        """Calculate angular error (via cosine similarity)."""
        return np.mean(self._angular_error(y_pred, y))

    def visualize(self, X: np.array, y: np.array, y_pred: np.array) -> None:
        """Visualize errors of neural network on given data."""
        if self.img_visualize:
            nr, nc = 1, 12
            n = nr * nc
            fig = plt.figure(figsize=(12, 2.))
            for i, (image, label, prediction) in enumerate(zip(X[:n], y[:n], y_pred)):
                plt.subplot(nr, nc, i + 1)
                plt.imshow(image, cmap='gray')
                error = self._angular_error(prediction.reshape(1, 2), label.reshape(1, 2))
                plt.title('%.1f' % error, color='g' if error < 7.0 else 'r')
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
            plt.tight_layout(pad=0.0)
            plt.show()

    def _angular_error(self, X, y):
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
