package github.umer0586.sensorserver.videostream

import android.content.Context
import android.util.Log
import org.java_websocket.WebSocket
import org.java_websocket.exceptions.WebsocketNotConnectedException
import org.java_websocket.handshake.ClientHandshake
import org.java_websocket.server.WebSocketServer
import java.net.InetSocketAddress
import java.nio.ByteBuffer

data class VideoServerInfo(val ipAddress: String, val port: Int)

class VideoWebSocketServer(private val context: Context, address: InetSocketAddress) :
    WebSocketServer(address) {

    private var onStartCallBack: ((VideoServerInfo) -> Unit)? = null
    private var onStopCallBack: (() -> Unit)? = null
    private var onErrorCallBack: ((Exception?) -> Unit)? = null
    private var connectionsChangeCallBack: ((List<WebSocket>) -> Unit)? = null

    private var serverStartUpFailed = false

    var isRunning = false
        private set

    companion object {
        private val TAG: String = VideoWebSocketServer::class.java.simpleName
        private const val CONNECTION_PATH_VIDEO = "/video"

        // WebSocket close codes
        const val CLOSE_CODE_UNSUPPORTED_REQUEST = 4002
        const val CLOSE_CODE_SERVER_STOPPED = 4004
        const val CLOSE_CODE_PERMISSION_DENIED = 4009
    }

    override fun onOpen(webSocket: WebSocket, handshake: ClientHandshake) {
        Log.i(TAG, "New connection ${webSocket.remoteSocketAddress} Resource descriptor: ${webSocket.resourceDescriptor}")

        val path = handshake.resourceDescriptor.lowercase()
        
        if (path.startsWith(CONNECTION_PATH_VIDEO)) {
            // Client is requesting video stream
            notifyConnectionsChanged()
        } else {
            webSocket.close(CLOSE_CODE_UNSUPPORTED_REQUEST, "Unsupported request")
        }
    }

    override fun onClose(webSocket: WebSocket, code: Int, reason: String, remote: Boolean) {
        Log.i(TAG, "Connection closed ${webSocket.remoteSocketAddress} with exit code $code additional info: $reason")
        notifyConnectionsChanged()
    }

    override fun onMessage(webSocket: WebSocket, message: String) {
        // Handle text messages from clients if needed
        Log.d(TAG, "Received text message: $message")
    }

    override fun onMessage(webSocket: WebSocket, message: ByteBuffer) {
        // Handle binary messages from clients if needed
        Log.d(TAG, "Received binary message of size: ${message.remaining()} bytes")
    }

    override fun onError(webSocket: WebSocket?, ex: Exception) {
        if (webSocket != null) {
            Log.e(TAG, "An error occurred on connection ${webSocket.remoteSocketAddress}", ex)
        } else {
            // Server-level error
            Log.e(TAG, "Server error", ex)
            onErrorCallBack?.invoke(ex)
            serverStartUpFailed = true
        }
        ex.printStackTrace()
    }

    override fun onStart() {
        onStartCallBack?.invoke(VideoServerInfo(address.hostName, port))
        isRunning = true
        Log.i(TAG, "Video server started successfully $address")
    }

    @kotlin.Throws(InterruptedException::class)
    override fun stop() {
        closeAllConnections()
        super.stop()
        Log.d(TAG, "stop() called")

        onStopCallBack?.run {
            if (!serverStartUpFailed) {
                invoke()
            }
        }

        isRunning = false
    }

    override fun run() {
        // Run the WebSocket server in a separate thread
        Thread { super.run() }.start()
    }

    /**
     * Send a video frame to all connected clients
     */
    fun sendVideoFrame(frameData: ByteArray) {
        if (connections.isEmpty()) {
            return
        }

        val buffer = ByteBuffer.wrap(frameData)
        
        for (webSocket in connections) {
            try {
                webSocket.send(buffer)
            } catch (e: WebsocketNotConnectedException) {
                e.printStackTrace()
            }
        }
    }

    private fun notifyConnectionsChanged() {
        Log.d(TAG, "notifyConnectionsChanged(): ${connections.size}")
        connectionsChangeCallBack?.invoke(ArrayList(connections))
    }

    private fun closeAllConnections() {
        connections.forEach { webSocket ->
            webSocket.close(CLOSE_CODE_SERVER_STOPPED, "Server stopped")
        }
    }

    fun onStart(callBack: ((VideoServerInfo) -> Unit)?) {
        onStartCallBack = callBack
    }

    fun onStop(callBack: (() -> Unit)?) {
        onStopCallBack = callBack
    }

    fun onError(callBack: ((Exception?) -> Unit)?) {
        onErrorCallBack = callBack
    }

    fun onConnectionsChange(callBack: ((List<WebSocket>) -> Unit)?) {
        connectionsChangeCallBack = callBack
    }
}
