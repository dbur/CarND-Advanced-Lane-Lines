# please read writeup.pdf
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

def calibrate_camera(calibration_img_glob):
    # calibrate camera
    # using 6x9 chessboard

    objpoints = [] # 3d points in real world
    imgpoints = [] # 2d points in image

    # real world points (0,0,0),(1,0,0)...(8,5,0)
    objp = np.zeros((9*6,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # for x,y coords

    images = glob.glob(calibration_img_glob) 
    gray = None
    # iterate through pictures of chess board to collect points for calibration
    for f in images:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    # use collected image and object poitns to calibrate camera
    ret,mtx,dist,rvecs,tvects = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

    # undistorting image test:
    # img = cv2.imread('camera_cal\\calibration3.jpg')
    # dst = cv2.undistort(img,mtx,dist,None,mtx)
    # cv2.imwrite('output_images\\calibration3-undistorted.jpg',dst)
    return mtx, dist


def color_gradient_thresh(img, s_thresh=(125,255), sx_thresh=(50,170), h_thresh=(0,30), r_thresh=(230,255)):
    img = np.copy(img)
    # convert to HLS, use S for lane lines
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS).astype(np.float)

    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    h_channel = hls[:,:,0]
    # BGR
    r_channel = img[:,:,2]

    sx = cv2.Sobel(l_channel,cv2.CV_64F,1,0)
    abs_sx = np.absolute(sx)
    scaled_sx = np.uint8(255*abs_sx / np.max(abs_sx))

    # threshold gradient
    sx_mask = np.zeros_like(s_channel)
    sx_mask[(scaled_sx>sx_thresh[0])&(scaled_sx<=sx_thresh[1])] = 1

    # threshold color
    s_mask = np.zeros_like(s_channel)
    s_mask[(s_channel>s_thresh[0])&(s_channel<=s_thresh[1])] = 1
    
    h_mask = np.zeros_like(h_channel)
    h_mask[(h_channel>h_thresh[0])&(h_channel<=h_thresh[1])] = 1

    r_mask = np.zeros_like(s_channel)
    r_mask[(r_channel>r_thresh[0])&(r_channel<=r_thresh[1])] = 1

    # where s and r overlap, or sx is sufficient
    intersecting_masks = ((s_mask==1) & (r_mask==1)) | (sx_mask==1) | ((h_mask==1) & (s_mask==1)) | ((h_mask==1) & (r_mask==1))
    int_mask = np.zeros_like(s_channel)
    int_mask[intersecting_masks] = 1
    return int_mask

def warp(img):
    # perspective transform
    img_size = (img.shape[1],img.shape[0])

    # points in source
    # handpicked by looking at some images
    # there must be a more dynamic way...
    src = np.float32([[256,683],[1048,683],[600,449],[686,449]])

    # points in destination
    dst = np.float32([[300,719],[1000,719],[300,0],[1000,0]])

    M = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    return warped

def unwarp(img):
    # perspective transform
    img_size = (img.shape[1],img.shape[0])

    # points in source
    # handpicked by looking at some images
    # there must be a more dynamic way...
    src = np.float32([[256,683],[1048,683],[600,449],[686,449]])

    # points in destination
    dst = np.float32([[300,719],[1000,719],[300,0],[1000,0]])

    M = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    return warped


def get_polyfit(birdview_img,imgname):
    hist = np.sum(birdview_img[int(birdview_img.shape[0]/2):,:],axis=0)
    midpoint = np.int(hist.shape[0]/2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:])+midpoint
    nwindows = 9
    window_height = np.int(birdview_img.shape[0]/nwindows)
    nonzero = birdview_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    out_img = (np.dstack((birdview_img, birdview_img, birdview_img))*255).astype(np.uint8)

    for window in range(nwindows):
        win_y_low = birdview_img.shape[0] - (window+1)*window_height
        win_y_high = birdview_img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit_meters = get_fit_in_meters(lefty, leftx, 2)
    right_fit_meters = get_fit_in_meters(righty, rightx, 2)

    ploty = np.linspace(0, birdview_img.shape[0]-1, birdview_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    # plt.savefig('output_images\\'+imgname)
    plt.clf()
    return left_fit, right_fit, left_fit_meters, right_fit_meters


# def get_polyfit_known_fit(birdview_img, left_fit, right_fit):
#     # not implemented
#     nonzero = birdview_img.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#     margin = 100
#     left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
#     right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
#     leftx = nonzerox[left_lane_inds]
#     lefty = nonzeroy[left_lane_inds] 
#     rightx = nonzerox[right_lane_inds]
#     righty = nonzeroy[right_lane_inds]
#     # Fit a second order polynomial to each
#     left_fit = np.polyfit(lefty, leftx, 2)
#     right_fit = np.polyfit(righty, rightx, 2)
#     left_fit_meters = get_fit_in_meters(lefty, leftx, 2)
#     right_fit_meters = get_fit_in_meters(righty, rightx, 2)
#     # Generate x and y values for plotting
#     # ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
#     # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#     # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#     return left_fit, right_fit, left_fit_meters, right_fit_meters


def get_fit_in_meters(y,x,degrees):
    ym_per_pix = get_ym_per_pix()
    xm_per_pix = get_xm_per_pix()
    return np.polyfit(y*ym_per_pix,x*xm_per_pix,degrees)

def get_xm_per_pix():
    # meters per pixel in x dimension
    return 3.7/700

def get_ym_per_pix():
    # meters per pixel in y dimension
    return 30/720.0

def get_curvature_meters(left_fit, right_fit, img_max_height):
    # point to evaluate at
    y_eval=img_max_height*get_ym_per_pix()

    left_curve = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curve = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curve, right_curve

def get_lane_center(left_fit,right_fit,img_max_height):
    left_x = left_fit[0]*img_max_height**2 + left_fit[1]*img_max_height + left_fit[2]
    right_x = right_fit[0]*img_max_height**2 + right_fit[1]*img_max_height + right_fit[2]
    return (left_x+right_x)/2.0

def get_center_offset_meters(lane_center,img_center):
    return (img_center-lane_center) * get_xm_per_pix()

def draw_lane(img, left_fit, right_fit):
    warp_zero = np.zeros_like(img[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # print(pts)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = unwarp(color_warp)

    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result


# test pipeline on images
def pipeline_test():
    # 1) calibrate camera
    mtx, dist = calibrate_camera('camera_cal/calibration*.jpg')
    test_images = glob.glob('test_images/*.jpg')
    for i in test_images:
        img = cv2.imread(i)
        # 2) undistort images
        undistorted_img = cv2.undistort(img,mtx,dist,None,mtx)
        # cv2.imwrite('output_images\\'+i,undistorted_img)
        # 3) threshold color and gradient
        thresholded_img = color_gradient_thresh(undistorted_img)
        # cv2.imwrite('output_images\\'+i,thresholded_img*255)
        birdview_img = warp(thresholded_img)
        # cv2.imwrite('output_images\\'+i,birdview_img*255)
        # 4) fit a polynomial
        left_fit, right_fit, lf_m, rf_m = get_polyfit(birdview_img,i)
        c = get_curvature_meters(lf_m,rf_m, 719)
        o = get_center_offset_meters(get_lane_center(left_fit,right_fit,birdview_img.shape[0]-1),birdview_img.shape[1]/2.0)
        img_with_lane = draw_lane(img,left_fit, right_fit)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_with_lane,'L Curvature: '+str(c[0])+' m',(20,100), font, 1, (255,255,255),1,cv2.LINE_AA)
        cv2.putText(img_with_lane,'R Curvature: '+str(c[1])+' m',(20,150), font, 1, (255,255,255),1,cv2.LINE_AA)
        if o < 0:
            cv2.putText(img_with_lane,'Meters Left of center :'+str(abs(o)),(20,200), font, 1, (255,255,255),1,cv2.LINE_AA)
        if o > 0:
            cv2.putText(img_with_lane,'Meters Right of center :'+str(abs(o))+' m right of center',(20,200), font, 1, (255,255,255),1,cv2.LINE_AA)
        # cv2.imwrite('output_images\\'+i,img_with_lane)

def process_video(video_in,video_out):
    mtx, dist = calibrate_camera('camera_cal/calibration*.jpg')
    clip1 = VideoFileClip(video_in, audio=False)
    lanes_clip = clip1.fl_image(lambda x: process_video_frames(x,mtx,dist))
    lanes_clip.write_videofile(video_out, audio=False)
    return

def process_video_frames(image, mtx, dist):
    # pipeline expects BGR so...
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    undistorted_img = cv2.undistort(image_bgr,mtx,dist,None,mtx)
    thresholded_img = color_gradient_thresh(undistorted_img)
    birdview_img = warp(thresholded_img)
    left_fit, right_fit, lf_m, rf_m = get_polyfit(birdview_img,'video_file')
    c = get_curvature_meters(lf_m,rf_m, 719)
    o = get_center_offset_meters(get_lane_center(left_fit,right_fit,birdview_img.shape[0]-1),birdview_img.shape[1]/2.0)
    # convert it back
    img_with_lane = draw_lane(image,left_fit, right_fit)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_with_lane,'L Curvature: '+str(c[0])+' m',(20,100), font, 1, (255,255,255),1,cv2.LINE_AA)
    cv2.putText(img_with_lane,'R Curvature: '+str(c[1])+' m',(20,150), font, 1, (255,255,255),1,cv2.LINE_AA)
    if o < 0:
        cv2.putText(img_with_lane,'Meters Left of center :'+str(abs(o)),(20,200), font, 1, (255,255,255),1,cv2.LINE_AA)
    if o > 0:
        cv2.putText(img_with_lane,'Meters Right of center :'+str(abs(o))+' m right of center',(20,200), font, 1, (255,255,255),1,cv2.LINE_AA)
    return img_with_lane

if __name__ == '__main__':
    print('start')
    print('test images...')
    # pipeline_test()
    print('video 1...')
    # process_video('project_video.mp4','project_video_output.mp4')
    print('video 2...')
    # process_video('challenge_video.mp4','challenge_video_output.mp4')
    print('video 3...')
    # process_video('harder_challenge_video.mp4','harder_challenge_video_output.mp4')
    print('done')