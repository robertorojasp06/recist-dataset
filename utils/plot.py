import numpy as np
import matplotlib.pyplot as plt
import cv2


def overlay_segmentation(image, mask, color=(1, 1, 0), alpha=0.5,
                         only_contours=False, contour_thickness=1):
    """
    Overlays a binary segmentation mask onto a grayscale image
    as an RGB image.

    Parameters:
    -----------
        image (ndarray): 2D grayscale image.
        mask (ndarray): 2D binary segmentation mask (same shape as image).
        color (tuple): RGB color for the mask overlay (default is yellow).
        alpha (float): Transparency of the mask overlay (0 = transparent, 1 = solid).

    Returns:
    --------
        overlay (ndarray): RGB image with segmentation overlay.
    """
    # Normalize the grayscale image to [0,1] for proper display
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    # Convert grayscale to RGB
    if image.ndim == 2:
        overlay = np.dstack([image] * 3)  # Shape (H, W, 3)
    elif image.ndim == 3:
        overlay = image.copy()
    else:
        raise ValueError("image must have shape (H, W) or (H, W, 3).")
    # Create a color mask
    color_mask = np.zeros_like(overlay)
    for i in range(3):  # Apply color to each channel
        color_mask[..., i] = color[i]
    # Blend the overlay with the image
    if only_contours:
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8) * 255,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        mask = cv2.cvtColor(
            np.zeros_like(mask).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
        cv2.drawContours(
            mask,
            contours,
            -1,
            (255, 255, 255),
            thickness=contour_thickness
        )
        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_BGR2GRAY
        )
    mask = mask.astype(bool)
    overlay = np.where(mask[..., None], (1 - alpha) * overlay + alpha * color_mask, overlay)
    return overlay


def overlay_masks(roi_grayscale, predicted_mask, gt_mask,
                  predicted_color=(0, 0, 1), gt_color=(1, 1, 0),
                  alpha=0.5):
    # Normalize image to the range [0, 1]
    roi_grayscale = (roi_grayscale - roi_grayscale.min()) / (roi_grayscale.max() - roi_grayscale.min()).astype(np.float32)
    # RGB image
    if roi_grayscale.ndim == 2:
        roi_3c = np.stack([roi_grayscale] * 3, axis=-1)
    else:
        roi_3c = roi_grayscale.copy()
    # Get color images
    predicted_mask = predicted_mask.astype('bool')
    gt_mask = gt_mask.astype('bool')
    predicted_color_mask = np.zeros_like(roi_3c)
    gt_color_mask = np.zeros_like(roi_3c)
    for i in range(3):  # Apply color to each channel
        predicted_color_mask[..., i] = predicted_color[i]
    for i in range(3):  # Apply color to each channel
        gt_color_mask[..., i] = gt_color[i]
    # Overlay masks
    overlay = np.where(
        gt_mask[..., None],
        (1 - alpha) * roi_3c + alpha * gt_color_mask,
        roi_3c
    )
    overlay = np.where(
        predicted_mask[..., None],
        (1 - alpha) * overlay + alpha * predicted_color_mask,
        overlay
    )
    return overlay


def show_box(box, ax, scaling=1.0, edgecolor=(0, 0, 1),
             linewidth=2):
    box = np.array([
        box[0] * (2.0 - scaling),
        box[1] * (2.0 - scaling),
        box[2] * scaling,
        box[3] * scaling
    ]).astype('int')
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            w,
            h,
            edgecolor=edgecolor,
            facecolor=(0, 0, 0, 0),
            lw=linewidth
        )
    )


def show_obb_opencv(center_row, center_col, width, height,
                    angle_deg, ax, edgecolor=(0, 0, 1),
                    linewidth=2):
    rect = (
        (center_col, center_row),
        (width, height),
        angle_deg
    )
    box = np.array(cv2.boxPoints(rect), dtype=np.int32)
    for j in range(4):
        ax.plot(
            [box[j][0], box[(j+1)%4][0]],
            [box[j][1], box[(j+1)%4][1]],
            color=edgecolor,
            linewidth=linewidth
        )


def show_ellipse_scikit(centroid_row, centroid_col, major_axis_length,
                        minor_axis_length, orientation, ax,
                        edgecolor=(0, 0, 1), linewidth=2):
    """Plot the ellipse according to the parameters given by regionprops
    from scikit image."""
    # Conversion from regionprops scikit parameters
    xc, yc = centroid_col, centroid_row
    a = minor_axis_length
    b = major_axis_length
    theta = -orientation
    # Parametric points
    t = np.linspace(0, 2 * np.pi, 100)
    x = xc + a / 2 * np.cos(t) * np.cos(theta) - b / 2 * np.sin(t) * np.sin(theta)
    y = yc + a / 2 * np.cos(t) * np.sin(theta) + b / 2 * np.sin(t) * np.cos(theta)

    ax.plot(
        x,
        y,
        color=edgecolor,
        linewidth=linewidth
    )


def plot_parametric_ellipse(xc, yc, a, b, theta_deg, ax=None,
                            edgecolor=(0, 0, 1), linewidth=2):
    """ Plot the ellipse specified the input parameters.

    Parameters
    ----------
        xc : center x-coordinate relative to bottom-left corner (positive to right)
        yc : center y-coordinate relative to bottom-left corner (positive to up)
        a  : major axis
        b  : minor axis
        theta_deg : angle in degrees between the positive x-axis and the major axis, measured conterclockwise.
    """
    # Parametric points
    theta = np.deg2rad(theta_deg)
    t = np.linspace(0, 2 * np.pi, 100)
    x = xc + a / 2 * np.cos(t) * np.cos(theta) - b / 2 * np.sin(t) * np.sin(theta)
    y = yc + a / 2 * np.cos(t) * np.sin(theta) + b / 2 * np.sin(t) * np.cos(theta)
    if ax:
        ax.plot(x, y, color=edgecolor, linewidth=linewidth)
    else:
        plt.plot(x, y, color=edgecolor, linewidth=linewidth)
        plt.show()
