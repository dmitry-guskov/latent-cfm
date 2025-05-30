import torch
from torchdyn.datasets import ToyDataset
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def eight_gaussians(n, dim=2, scale=1, var=1):
    if dim < 2:
        raise ValueError("dim must be at least 2")

    centers_2d = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1/np.sqrt(2), 1/np.sqrt(2)),
        (1/np.sqrt(2), -1/np.sqrt(2)),
        (-1/np.sqrt(2), 1/np.sqrt(2)),
        (-1/np.sqrt(2), -1/np.sqrt(2))
    ]
    centers_2d = torch.tensor(centers_2d) * scale  # Shape (8, 2)
    
    centers = torch.zeros(8, dim)
    centers[:, :2] = centers_2d  # Higher dimensions remain zero

    m = torch.distributions.MultivariateNormal(
        loc=torch.zeros(dim),
        covariance_matrix=var * torch.eye(dim)
    )


    labels = torch.multinomial(torch.ones(8), n, replacement=True)  # Shape (n,)
    
    # Select centers corresponding to labels
    selected_centers = centers[labels]  # Shape (n, dim)
    
    noise = m.sample((n,))
    data = selected_centers + noise
    
    return data, labels


def make_text_dataset(word,
                      n_samples=500,
                      font_path=None,
                      font_size=200,
                      noise_sigma=0.02,
                      char_spacing: int = 1,
                     normalize = True):
    """
    Create a dataset of points sampled inside each letter of `word`.
    Returns a Dataset yielding (point, letter_index).
    - char_spacing: extra pixels inserted between letters
    """
    # 1) Load font
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    n_samples_per_letter = n_samples // len(word)
    # 2) Compute per-character sizes & total width with spacing
    dummy = Image.new('L', (1,1), 0)
    dr    = ImageDraw.Draw(dummy)
    char_sizes = [dr.textbbox((0,0), c, font=font)[2:4] for c in word]
    # char_sizes[i] = (width, height)
    heights = [h for (_,h) in char_sizes]
    max_h = max(heights)
    total_w = sum(w for (w,_) in char_sizes) + char_spacing * (len(word)-1)

    # 3) Sample points per letter
    all_points = []
    all_labels = []
    x_offset = 0

    for idx, (c, (cw, ch)) in enumerate(zip(word, char_sizes)):
        # Render letter c in its own image
        img = Image.new('L', (total_w, max_h), 0)
        draw = ImageDraw.Draw(img)
        draw.text((x_offset, 0), c, fill=255, font=font)
        mask = np.array(img) > 128

        ys, xs = np.where(mask)
        coords = np.stack([xs, -ys], axis=1).astype(np.float32)
        if coords.shape[0] == 0:
            x_offset += cw + char_spacing
            continue

        # Uniformly sample inside this letter
        picks = np.random.choice(coords.shape[0], n_samples_per_letter, replace=True)
        pts = coords[picks]  # shape (n_samples_per_letter, 2)

        # Add some jitter
        pts += np.random.randn(*pts.shape) * noise_sigma

        # Normalize to [-1,1]^2
        if normalize:
        # xs in [0, total_w], ys in [0, max_h]
            pts[:,0] = (pts[:,0] / (total_w/2)) - 1.0
            pts[:,1] = (pts[:,1] / (max_h/2)) - 1.0

        all_points.append(pts)
        all_labels.append(np.full(n_samples_per_letter, idx, dtype=np.int64))

        x_offset += cw + char_spacing

    # 4) Stack and shuffle
    points = np.vstack(all_points)
    labels = np.hstack(all_labels)
    perm   = np.random.permutation(points.shape[0])
    points = torch.from_numpy(points[perm]).float()
    labels = torch.from_numpy(labels[perm]).long()

    return points, labels