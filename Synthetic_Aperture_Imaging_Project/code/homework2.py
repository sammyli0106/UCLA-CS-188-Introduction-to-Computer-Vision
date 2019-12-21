import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import match_template, peak_local_max


def processVideo(path):
    # loads video from given path
    # returns video frames in two arrays, one color & one grayscale

    video = cv2.VideoCapture(path)
    success = True
    gray_result, color_result = [], []

    while success:
        success, frame = video.read()
        if frame is not None:  # for last frame & weird frames
            color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            color_result.append(color_image)

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_result.append(gray_image)

    video.release()
    return gray_result, color_result


def getTemplate(image, x, w, y, h):
    # crop the template image using bottom left coordinates
    return image[y:y+h, x:x+w].copy()


def getPeak(image, template):
    # for a single image, match the template
    # returns the coordinates of the maximum correlation in the matched region
    result = match_template(image, template, pad_input=True)
    peak = peak_local_max(result, min_distance=10, threshold_rel=0.5, num_peaks=1)
    return peak[0][1], peak[0][0]


if __name__ == "__main__":

    # set plot_flag to 1 to plot intermediate steps, save_flag to 1 to save final image
    plot_flag = 1
    save_flag = 0
    save_path = "/Users/insiya/Desktop/Homework2/"  # can be blank if save_flag is 0

    #  X & Y coordinates are the top left corner of template bounding box

    templateX, templateW, templateY, templateH = 350, 50, 180, 60  # video.mp4, terrarium <- best
    # templateX, templateW, templateY, templateH = 425, 70, 280, 70  # video.mp4, salt & pepper shakers
    # templateX, templateW, templateY, templateH = 485, 50, 250, 50  # video.mp4, green glass
    # templateX, templateW, templateY, templateH = 275, 50, 245, 85  # video.mp4, birthday mug <- DO NOT USE!

    # windowX, windowW, windowY, windowH = ... # if we wanted to use a window

    print("Processing video and retrieving template...")
    gray_images, color_images = processVideo('/Users/insiya/Desktop/Homework2/video.mp4')
    template = getTemplate(gray_images[0], templateX, templateW, templateY, templateH)

    # print(len(color_images)) # get number of frames

    # plot rectangle surrounding template on first image
    if plot_flag:
        first_frame = np.array(color_images[0], dtype=np.uint8)
        plt.imshow(first_frame)
        plt.title('Template in First Frame', fontsize=12)
        ax = plt.gca()
        # patches.Rectangle uses bottom left corner instead of top left corner
        template_rect = patches.Rectangle((templateX, templateY), templateW, templateH, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(template_rect)
        plt.show()

    # plot a sample cross correlation coefficient matrix
    if plot_flag:
        sample_matrix = match_template(gray_images[1], template, pad_input=True)
        peaks = peak_local_max(sample_matrix, min_distance=10, threshold_rel=0.5, num_peaks=1)
        plt.imshow(sample_matrix, cmap='gray')
        plt.plot(peaks[:, 1], peaks[:, 0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
        plt.title('Sample Cross Correlation Coefficient Matrix', fontsize=12)
        plt.xlabel('Pixel location in X Direction', fontsize=10)
        plt.ylabel('Pixel location in Y Direction', fontsize=10)
        plt.colorbar()
        plt.show()

    # get peaks of normalized cross correlation operation
    print("Using template matching to find maximums...")
    co_peaks_x, co_peaks_y = [], []
    for g in gray_images:
        x, y = getPeak(g, template)
        co_peaks_x.append(x)
        co_peaks_y.append(y)

    # plot graph of pixel shifts
    if plot_flag:
        plt.scatter(np.array(co_peaks_x), np.array(co_peaks_y))
        plt.title('X Pixel Shift vs. Y Pixel Shift', fontsize=12)
        plt.xlabel('X Pixel Shift', fontsize=10)
        plt.ylabel('Y Pixel Shift', fontsize=10)
        plt.show()

    # shift each image to superimpose each other
    print("Shifting images... ")
    for i in range(len(color_images)):
        num_rows, num_cols = color_images[i].shape[:2]

        trans_x = co_peaks_x[0] - co_peaks_x[i]
        trans_y = co_peaks_y[0] - co_peaks_y[i]

        translation_matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        color_images[i] = cv2.warpAffine(color_images[i], translation_matrix, (num_cols, num_rows))

    # show a sample translated image
    if plot_flag:
        plt.imshow(color_images[50])
        plt.title('Sample Translated Image, Frame 50', fontsize=12)
        plt.show()

    # averaging step
    print("Producing bokeh effect...")
    color_images = np.array(color_images, dtype=np.float32)
    color_images[color_images == 0] = np.nan  # ignore zeros while averaging
    final = np.nanmean(color_images, axis=0)
    final = np.array(final, dtype=np.uint8)

    # plot final bokeh filter image
    plt.imshow(final)
    plt.title('Synthesized Bokeh Filter Image', fontsize=12)
    plt.show()

    if save_flag:
        cv2.imwrite(save_path + "bokeh.png", cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

