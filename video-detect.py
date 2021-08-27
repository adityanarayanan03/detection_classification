import jetson.utils
import cv2

camera = jetson.utils.videoSource("detection_classification/test_videos/1.mp4")
display = jetson.utils.videoOutput("display://0")


import jetson.inference
net = jetson.inference.detectNet("ssd-mobilenet-v2")


while display.IsStreaming():
	img = camera.Capture()
	#print(type(img))
	net.Detect(img)
	bgr_img = jetson.utils.cudaAllocMapped(width=img.width,height=img.height, format='bgr8')
	jetson.utils.cudaConvertColor(img, bgr_img)
	jetson.utils.cudaDeviceSynchronize()
	cv_img = jetson.utils.cudaToNumpy(bgr_img)
	cv2.imshow('frame', cv_img)
	cv2.waitKey(1)
	#display.Render(img)
	#print(type(img))
	#display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
	#csv_writer.writerow([net.GetNetworkTime()])
	#net.PrintProfilerTimes()

