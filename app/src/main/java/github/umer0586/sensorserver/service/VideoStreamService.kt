package github.umer0586.sensorserver.service

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.util.Log
import android.view.Surface
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import github.umer0586.sensorserver.R
import github.umer0586.sensorserver.activities.MainActivity
import github.umer0586.sensorserver.broadcastreceiver.BroadcastMessageReceiver
import github.umer0586.sensorserver.customextensions.getHotspotIp
import github.umer0586.sensorserver.customextensions.getIp
import github.umer0586.sensorserver.setting.AppSettings
import github.umer0586.sensorserver.videostream.VideoServerInfo
import github.umer0586.sensorserver.videostream.VideoWebSocketServer
import java.net.InetSocketAddress
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

interface VideoServerStateListener {
    fun onStart(videoServerInfo: VideoServerInfo)
    fun onStop()
    fun onError(exception: Exception?)
    fun onRunning(videoServerInfo: VideoServerInfo)
}

class VideoStreamService : Service() {
    
    private var videoServerStateListener: VideoServerStateListener? = null
    private var videoWebSocketServer: VideoWebSocketServer? = null
    private lateinit var broadcastMessageReceiver: BroadcastMessageReceiver
    private lateinit var appSettings: AppSettings
    
    // Camera related variables
    private lateinit var cameraManager: CameraManager
    private var cameraId: String = "0" // Default to first camera
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private lateinit var imageReader: ImageReader
    private lateinit var backgroundThread: HandlerThread
    private lateinit var backgroundHandler: Handler
    private val cameraOpenCloseLock = Semaphore(1)
    
    companion object {
        private val TAG: String = VideoStreamService::class.java.simpleName
        const val CHANNEL_ID = "VideoStreamServiceChannel"
        const val ON_GOING_NOTIFICATION_ID = 333
        val ACTION_STOP_VIDEO_STREAM = "ACTION_STOP_VIDEO_STREAM_" + VideoStreamService::class.java.name
        private const val DEFAULT_VIDEO_WIDTH = 640
        private const val DEFAULT_VIDEO_HEIGHT = 480
        private const val DEFAULT_PORT = 8086
    }
    
    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate() - Initializing VideoStreamService")
        
        try {
            // Initialize camera manager
            cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            Log.d(TAG, "CameraManager initialized")
            
            // Initialize app settings
            appSettings = AppSettings(this)
            Log.d(TAG, "AppSettings initialized")
            
            // Create notification channel for Android 8.0+
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                createNotificationChannel()
                Log.d(TAG, "Notification channel created")
            }
            
            // Register broadcast receiver for stopping the service
            broadcastMessageReceiver = BroadcastMessageReceiver(applicationContext)
            broadcastMessageReceiver.setOnMessageReceived { intent ->
                if (intent.action == ACTION_STOP_VIDEO_STREAM) {
                    Log.d(TAG, "Received broadcast to stop video stream")
                    stopSelf()
                }
            }
            
            val intentFilter = IntentFilter().apply {
                addAction(ACTION_STOP_VIDEO_STREAM)
            }
            
            registerReceiver(broadcastMessageReceiver, intentFilter)
            Log.d(TAG, "BroadcastMessageReceiver registered")
            
            // Start background thread for camera operations
            startBackgroundThread()
            Log.d(TAG, "Background thread started")
            
            Log.d(TAG, "VideoStreamService created successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error in onCreate: ${e.message}", e)
        }
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand() - Starting VideoStreamService")
        
        try {
            // Get IP address
            val wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as android.net.wifi.WifiManager
            val ipAddress = if (appSettings.isAllInterfaceOptionEnabled()) {
                Log.d(TAG, "Using all interfaces")
                "0.0.0.0"
            } else {
                Log.d(TAG, "Using WiFi IP address")
                val ipInt = wifiManager.connectionInfo.ipAddress
                String.format("%d.%d.%d.%d",
                    ipInt and 0xff,
                    ipInt shr 8 and 0xff,
                    ipInt shr 16 and 0xff,
                    ipInt shr 24 and 0xff)
            }
            
            Log.d(TAG, "IP address: $ipAddress")
            
            if (ipAddress.isNullOrEmpty() || ipAddress == "0.0.0.0" && !appSettings.isAllInterfaceOptionEnabled()) {
                Log.e(TAG, "Failed to get IP address")
                videoServerStateListener?.onError(Exception("Failed to get IP address"))
                stopForegroundCompat(true)
                return START_NOT_STICKY
            }
            
            // Get port from settings or use default
            val port = appSettings.getWebsocketPortNo()
            Log.d(TAG, "Using port: $port")
            
            try {
                // Initialize WebSocket server
                Log.d(TAG, "Initializing VideoWebSocketServer")
                val socketAddress = InetSocketAddress(ipAddress, port)
                videoWebSocketServer = VideoWebSocketServer(applicationContext, socketAddress)
                
                // Set up callbacks
                videoWebSocketServer?.onStart { serverInfo ->
                    Log.d(TAG, "VideoWebSocketServer started successfully")
                    videoServerStateListener?.onStart(serverInfo)
                    
                    // Start camera preview
                    setupCamera()
                    
                    // Create notification
                    val activityIntent = Intent(this, MainActivity::class.java)
                    val broadcastIntent = Intent(ACTION_STOP_VIDEO_STREAM).apply {
                        setPackage(packageName)
                    }
                    
                    val pendingIntentActivity = PendingIntent.getActivity(this, 0, activityIntent, PendingIntent.FLAG_IMMUTABLE)
                    val pendingIntentBroadcast = PendingIntent.getBroadcast(this, 0, broadcastIntent, PendingIntent.FLAG_IMMUTABLE)
                    
                    val notificationBuilder = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
                        .apply {
                            setSmallIcon(R.drawable.ic_radar_signal)
                            setContentTitle("Video Stream Running...")
                            setContentText("ws://$ipAddress:$port")
                            setPriority(NotificationCompat.PRIORITY_DEFAULT)
                            setContentIntent(pendingIntentActivity)
                            addAction(android.R.drawable.ic_lock_power_off, "stop", pendingIntentBroadcast)
                            setAutoCancel(false)
                        }
                    
                    val notification = notificationBuilder.build()
                    startForeground(ON_GOING_NOTIFICATION_ID, notification)
                }
                
                videoWebSocketServer?.onStop {
                    Log.d(TAG, "VideoWebSocketServer stopped")
                    videoServerStateListener?.onStop()
                    stopCamera()
                    stopForegroundCompat(true)
                }
                
                videoWebSocketServer?.onError { exception ->
                    Log.e(TAG, "VideoWebSocketServer error: ${exception?.message}", exception)
                    videoServerStateListener?.onError(exception)
                    stopCamera()
                    stopForegroundCompat(true)
                }
                
                // Start the WebSocket server
                Log.d(TAG, "Starting VideoWebSocketServer")
                videoWebSocketServer?.run()
                
            } catch (e: Exception) {
                Log.e(TAG, "Error setting up VideoWebSocketServer: ${e.message}", e)
                videoServerStateListener?.onError(e)
                stopForegroundCompat(true)
                return START_NOT_STICKY
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in onStartCommand: ${e.message}", e)
            if (videoServerStateListener != null) {
                videoServerStateListener?.onError(e)
            }
            stopForegroundCompat(true)
            return START_NOT_STICKY
        }
        
        return START_NOT_STICKY
    }
    
    override fun onBind(intent: Intent): IBinder {
        Log.d(TAG, "onBind() called")
        return LocalBinder()
    }
    
    inner class LocalBinder : Binder() {
        // 只保留一种方式获取服务实例
        fun getService(): VideoStreamService = this@VideoStreamService
    }
    
    fun setServerStateListener(listener: VideoServerStateListener?) {
        this.videoServerStateListener = listener
        Log.d(TAG, "VideoServerStateListener set")
    }
    
    fun checkState(): VideoServerInfo? {
        Log.d(TAG, "checkState() called")
        return videoWebSocketServer?.let {
            if (it.isRunning) {
                val address = it.address
                VideoServerInfo(address.hostName, address.port)
            } else {
                null
            }
        }
    }
    
    private fun setupCamera() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "Camera permission not granted")
            videoServerStateListener?.onError(Exception("Camera permission not granted"))
            return
        }
        
        try {
            Log.d(TAG, "Setting up camera...")
            // Find the first back-facing camera
            val cameraIdList = cameraManager.cameraIdList
            Log.d(TAG, "Available cameras: ${cameraIdList.joinToString()}")
            
            if (cameraIdList.isEmpty()) {
                Log.e(TAG, "No cameras available on device")
                videoServerStateListener?.onError(Exception("No cameras available on device"))
                return
            }
            
            // 首先尝试找到后置摄像头
            for (cameraId in cameraIdList) {
                val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                    this.cameraId = cameraId
                    Log.d(TAG, "Found back-facing camera: $cameraId")
                    break
                }
            }
            
            // 如果没有找到后置摄像头，使用第一个可用的摄像头
            if (this.cameraId == null) {
                this.cameraId = cameraIdList[0]
                Log.d(TAG, "No back-facing camera found, using camera: ${this.cameraId}")
            }
            
            // Create ImageReader for processing frames
            Log.d(TAG, "Creating ImageReader with dimensions: ${DEFAULT_VIDEO_WIDTH}x${DEFAULT_VIDEO_HEIGHT}")
            imageReader = ImageReader.newInstance(
                DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT,
                ImageFormat.JPEG, 2
            )
            
            imageReader.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage()
                if (image != null) {
                    try {
                        val buffer = image.planes[0].buffer
                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)
                        
                        // Send the JPEG data to connected clients
                        videoWebSocketServer?.sendVideoFrame(bytes)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error processing image: ${e.message}", e)
                    } finally {
                        image.close()
                    }
                }
            }, backgroundHandler)
            
            // Open the camera
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw RuntimeException("Time out waiting to lock camera opening.")
            }
            
            Log.d(TAG, "Opening camera: $cameraId")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                try {
                    cameraManager.openCamera(cameraId!!, stateCallback, backgroundHandler)
                    Log.d(TAG, "Camera open request sent")
                } catch (e: CameraAccessException) {
                    Log.e(TAG, "Failed to open camera: ${e.message}", e)
                    videoServerStateListener?.onError(e)
                    cameraOpenCloseLock.release()
                } catch (e: Exception) {
                    Log.e(TAG, "Unexpected error opening camera: ${e.message}", e)
                    videoServerStateListener?.onError(e)
                    cameraOpenCloseLock.release()
                }
            } else {
                Log.e(TAG, "Camera permission not granted for openCamera")
                videoServerStateListener?.onError(Exception("Camera permission not granted for openCamera"))
                cameraOpenCloseLock.release()
            }
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to setup camera", e)
            videoServerStateListener?.onError(e)
        } catch (e: InterruptedException) {
            Log.e(TAG, "Interrupted while trying to lock camera", e)
            videoServerStateListener?.onError(e)
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error setting up camera: ${e.message}", e)
            videoServerStateListener?.onError(e)
        }
    }
    
    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            Log.d(TAG, "Camera opened successfully")
            cameraOpenCloseLock.release()
            cameraDevice = camera
            createCameraPreviewSession()
        }
        
        override fun onDisconnected(camera: CameraDevice) {
            Log.d(TAG, "Camera disconnected")
            cameraOpenCloseLock.release()
            camera.close()
            cameraDevice = null
        }
        
        override fun onError(camera: CameraDevice, error: Int) {
            Log.e(TAG, "Camera device error: $error")
            cameraOpenCloseLock.release()
            camera.close()
            cameraDevice = null
            videoServerStateListener?.onError(Exception("Camera error code: $error"))
        }
    }
    
    private fun createCameraPreviewSession() {
        try {
            Log.d(TAG, "Creating camera preview session")
            if (cameraDevice == null) {
                Log.e(TAG, "Camera device is null when creating preview session")
                videoServerStateListener?.onError(Exception("Camera device is null when creating preview session"))
                return
            }
            
            if (!::imageReader.isInitialized) {
                Log.e(TAG, "ImageReader is not initialized when creating preview session")
                videoServerStateListener?.onError(Exception("ImageReader is not initialized when creating preview session"))
                return
            }
            
            val surface = imageReader.surface
            Log.d(TAG, "Got surface from imageReader")
            
            // Set up capture request
            val captureRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
            if (captureRequestBuilder == null) {
                Log.e(TAG, "Failed to create capture request builder")
                videoServerStateListener?.onError(Exception("Failed to create capture request builder"))
                return
            }
            
            captureRequestBuilder.addTarget(surface)
            Log.d(TAG, "Added surface to capture request builder")
            
            // Create a capture session
            try {
                Log.d(TAG, "Creating capture session with surface")
                cameraDevice?.createCaptureSession(
                    listOf(surface),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            Log.d(TAG, "Camera capture session configured successfully")
                            captureSession = session
                            try {
                                // Auto-focus mode
                                captureRequestBuilder.set(
                                    CaptureRequest.CONTROL_AF_MODE,
                                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                                )
                                
                                // Set JPEG quality
                                captureRequestBuilder.set(
                                    CaptureRequest.JPEG_QUALITY,
                                    90.toByte() // 0-100
                                )
                                
                                // Start the preview
                                val captureRequest = captureRequestBuilder.build()
                                captureSession?.setRepeatingRequest(captureRequest, null, backgroundHandler)
                                Log.d(TAG, "Camera preview started successfully")
                                
                                // Notify that video stream is running
                                videoWebSocketServer?.let { server ->
                                    if (server.isRunning) {
                                        val address = server.address
                                        val serverInfo = VideoServerInfo(address.hostName, address.port)
                                        videoServerStateListener?.onRunning(serverInfo)
                                        Log.d(TAG, "Notified listeners that video stream is running")
                                    }
                                }
                            } catch (e: CameraAccessException) {
                                Log.e(TAG, "Failed to start camera preview: ${e.message}", e)
                                videoServerStateListener?.onError(e)
                            } catch (e: Exception) {
                                Log.e(TAG, "Unexpected error starting camera preview: ${e.message}", e)
                                videoServerStateListener?.onError(e)
                            }
                        }
                        
                        override fun onConfigureFailed(session: CameraCaptureSession) {
                            Log.e(TAG, "Failed to configure camera capture session")
                            videoServerStateListener?.onError(Exception("Failed to configure camera capture session"))
                        }
                    },
                    backgroundHandler
                )
                Log.d(TAG, "Capture session creation request sent")
            } catch (e: Exception) {
                Log.e(TAG, "Error creating capture session: ${e.message}", e)
                videoServerStateListener?.onError(e)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error in createCameraPreviewSession: ${e.message}", e)
            videoServerStateListener?.onError(e)
        }
    }
    
    private fun stopCamera() {
        try {
            Log.d(TAG, "Stopping camera")
            cameraOpenCloseLock.acquire()
            captureSession?.close()
            captureSession = null
            
            cameraDevice?.close()
            cameraDevice = null
            
            // 检查imageReader是否已初始化，避免UninitializedPropertyAccessException
            if (::imageReader.isInitialized) {
                imageReader.close()
            }
            Log.d(TAG, "Camera stopped")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping camera: ${e.message}", e)
        } finally {
            cameraOpenCloseLock.release()
        }
    }
    
    private fun startBackgroundThread() {
        Log.d(TAG, "Starting background thread")
        backgroundThread = HandlerThread("CameraBackground")
        backgroundThread.start()
        backgroundHandler = Handler(backgroundThread.looper)
    }
    
    private fun stopBackgroundThread() {
        Log.d(TAG, "Stopping background thread")
        try {
            backgroundThread.quitSafely()
            backgroundThread.join()
        } catch (e: InterruptedException) {
            Log.e(TAG, "Error stopping background thread: ${e.message}", e)
        }
    }
    
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val name = "Video Stream Service"
            val descriptionText = "Channel for Video Stream Service notifications"
            val importance = NotificationManager.IMPORTANCE_DEFAULT
            val channel = NotificationChannel(CHANNEL_ID, name, importance).apply {
                description = descriptionText
            }
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
            Log.d(TAG, "Notification channel created")
        }
    }
    
    private fun stopForegroundCompat(removeNotification: Boolean = true) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(if (removeNotification) Service.STOP_FOREGROUND_REMOVE else Service.STOP_FOREGROUND_DETACH)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(removeNotification)
        }
        Log.d(TAG, "Foreground service stopped")
    }
    
    override fun onDestroy() {
        Log.d(TAG, "onDestroy() - Cleaning up VideoStreamService")
        try {
            // Stop camera
            stopCamera()
            
            // Stop WebSocket server
            videoWebSocketServer?.stop()
            
            // Stop background thread
            stopBackgroundThread()
            
            // Unregister broadcast receiver
            unregisterReceiver(broadcastMessageReceiver)
            
            Log.d(TAG, "VideoStreamService destroyed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error in onDestroy: ${e.message}", e)
        } finally {
            super.onDestroy()
        }
    }
}
