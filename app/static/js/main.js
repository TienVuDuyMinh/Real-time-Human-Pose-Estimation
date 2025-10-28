// File: app/static/js/main.js 
document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById('video'); // <-- Sử dụng lại thẻ video
    const canvas = document.createElement('canvas'); // Canvas để gửi frame
    const context = canvas.getContext('2d');
    // const resultImage = document.getElementById('result-image'); // Không cần thẻ img nữa
    const actionText = document.getElementById('action-text');
    const confidenceText = document.getElementById('confidence-text');
    const socket = io();

    let sendInterval; // Biến để lưu interval

    socket.on('connect', () => {
        console.log('Đã kết nối tới server!');
    });

    // --- Chỉ nhận action và confidence ---
    socket.on('action_update', (data) => { // <-- Quay lại tên event cũ 'action_update'
        actionText.innerText = data.action;
        confidenceText.innerText = data.confidence;
        // Không cập nhật ảnh nữa
    });
    // ------------------------------------

    // Truy cập camera
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                video.srcObject = stream; // <-- Hiển thị stream trực tiếp trên thẻ video
                video.onloadedmetadata = () => {
                    // Chỉ bắt đầu gửi sau khi video đã sẵn sàng
                    startSendingFrames();
                };
            })
            .catch((error) => {
                console.error("Lỗi khi truy cập camera:", error);
            });
    } else {
        alert("Trình duyệt không hỗ trợ getUserMedia.");
    }

    function startSendingFrames() {
        if (sendInterval) clearInterval(sendInterval);

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        // Không cần set kích thước resultImage

        console.log(`Bắt đầu gửi frame (${canvas.width}x${canvas.height})`);

        // Gửi frame (có thể tăng lại FPS nếu muốn)
        sendInterval = setInterval(() => {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            socket.emit('video_frame', canvas.toDataURL('image/jpeg'));
        }, 1000 / 15); // <-- Tăng lại 15 FPS (Bạn có thể điều chỉnh)
    }

    // Dừng gửi frame khi đóng tab/trình duyệt
    window.addEventListener('beforeunload', () => {
        if (sendInterval) clearInterval(sendInterval);
        socket.disconnect();
    });
});