import cv2
import numpy as np

class Video(object):
    '''
    Wrapper for OpenCV video capture, support indexing and slicing
    '''

    def __init__(self, file_name):
        # initialize the video file
        self.file_name = file_name
        self.cap = cv2.VideoCapture(self.file_name)
        
        #load basic information of the video file
        self.num_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_ = self.cap.get(cv2.CAP_PROP_FPS)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self, i):
        # set the current frame position (index starting from zeros)
        if i>=0 and i<self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if ret == 1:
                return frame[:, :, 2::-1]
            else:
                raise Exception('Frame %d is corrupted.' % i)
        else:
            raise IndexError("Index %d is out of range of [0,%d)" % (i,self.total_frames))

    @property
    def total_frames(self):
        return self.num_frame
    
    @property
    def fps(self):
        return self.fps_
    
    @property
    def width(self):
        return self.w
    
    @property
    def height(self):
        return self.h
    
    @property
    def num_channels(self):
        return 3
    
    @property
    def shape(self):
        return (self.total_frames,self.height,self.width,self.num_channels)

    def __getitem__(self,val):
        # return a numpy array based on the index slicing information
        vid_ids = np.arange(0,self.total_frames)[val]
        if type(vid_ids) == np.ndarray:
            vid_data = np.zeros((len(vid_ids),self.height,self.width,self.num_channels))
            for i,frame_id in enumerate(vid_ids):
                vid_data[i,:,:,:] = self.read_frame(frame_id)
            return vid_data
        else:
            return self.read_frame(vid_ids)
        
    @property
    def data(self):
        # returns entire video as an numpy array, use with caution for large video files
        return self[0:self.total_frames]
        
        
    def __len__(self):
        return self.total_frames
    
    def __repr__(self):
        return 'Video(%s,%.1f,%d,%d,%d,%d)' % (self.file_name,self.fps,self.total_frames,
                                               self.height,self.width,self.num_channels)
    
    def __str__(self):
        return 'Video named %s, %.1f FPS, %d frames, %d height, %d width and %d colors' % (self.file_name,self.fps,self.total_frames,
                                                                                           self.height,self.width,self.num_channels)

    def close(self):
        self.cap.release()