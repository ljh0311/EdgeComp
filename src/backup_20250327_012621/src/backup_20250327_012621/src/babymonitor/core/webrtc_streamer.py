import cv2
import logging
import threading
import time
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame

logger = logging.getLogger(__name__)

class VideoStreamTrack(MediaStreamTrack):
    """
    A video stream track that returns frames from a camera.
    """
    kind = "video"

    def __init__(self):
        super().__init__()
        self.frame_queue = []
        self.queue_lock = threading.Lock()
        self.frame_count = 0
        self.last_frame = None
        self.stopped = False

    def add_frame(self, frame):
        """
        Add a frame to the queue.
        
        Args:
            frame: OpenCV frame (numpy array)
        """
        if frame is None:
            return
            
        with self.queue_lock:
            # Keep only the latest frame
            self.frame_queue = [frame]
            self.last_frame = frame

    async def recv(self):
        """
        Return a frame from the queue.
        """
        if self.stopped:
            raise Exception("Track has been stopped")
            
        # Get the latest frame
        frame = None
        with self.queue_lock:
            if self.frame_queue:
                frame = self.frame_queue.pop(0)
            elif self.last_frame is not None:
                # Reuse the last frame if no new frames
                frame = self.last_frame
                
        if frame is None:
            # Create a black frame if no frames are available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        # Convert to VideoFrame
        self.frame_count += 1
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self.frame_count * 1000  # milliseconds
        video_frame.time_base = "1/1000"
        
        return video_frame

    def stop(self):
        """
        Stop the track.
        """
        self.stopped = True
        super().stop()


class WebRTCStreamer:
    """
    WebRTC streamer for the baby monitor.
    """
    def __init__(self):
        self.peer_connections = {}
        self.video_track = VideoStreamTrack()
        self.ice_servers = [
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=["stun:stun1.l.google.com:19302"])
        ]
        self.rtc_config = RTCConfiguration(iceServers=self.ice_servers)
        self.relay = MediaRelay()
        
    def add_frame(self, frame):
        """
        Add a frame to the video track.
        
        Args:
            frame: OpenCV frame (numpy array)
        """
        self.video_track.add_frame(frame)
        
    async def create_offer(self, client_id):
        """
        Create a WebRTC offer for a client.
        
        Args:
            client_id: Client ID
            
        Returns:
            RTCSessionDescription: The offer
        """
        # Create a new peer connection
        pc = RTCPeerConnection(configuration=self.rtc_config)
        self.peer_connections[client_id] = pc
        
        # Add the video track
        pc.addTrack(self.relay.subscribe(self.video_track))
        
        # Create the offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        # Set up event handlers
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE connection state for client {client_id}: {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed" or pc.iceConnectionState == "closed":
                await self.close_peer_connection(client_id)
                
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state for client {client_id}: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await self.close_peer_connection(client_id)
                
        return pc.localDescription
        
    async def process_answer(self, client_id, answer):
        """
        Process a WebRTC answer from a client.
        
        Args:
            client_id: Client ID
            answer: RTCSessionDescription
        """
        if client_id not in self.peer_connections:
            logger.error(f"No peer connection for client {client_id}")
            return
            
        pc = self.peer_connections[client_id]
        await pc.setRemoteDescription(answer)
        
    async def add_ice_candidate(self, client_id, candidate):
        """
        Add an ICE candidate from a client.
        
        Args:
            client_id: Client ID
            candidate: RTCIceCandidate
        """
        if client_id not in self.peer_connections:
            logger.error(f"No peer connection for client {client_id}")
            return
            
        pc = self.peer_connections[client_id]
        await pc.addIceCandidate(candidate)
        
    async def close_peer_connection(self, client_id):
        """
        Close a peer connection.
        
        Args:
            client_id: Client ID
        """
        if client_id in self.peer_connections:
            pc = self.peer_connections[client_id]
            await pc.close()
            del self.peer_connections[client_id]
            
    async def close_all(self):
        """
        Close all peer connections.
        """
        for client_id in list(self.peer_connections.keys()):
            await self.close_peer_connection(client_id)
            
        # Stop the video track
        self.video_track.stop() 