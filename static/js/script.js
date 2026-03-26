document.addEventListener('DOMContentLoaded', function() {
    const cameraSource = document.getElementById('cameraSource');
    const ipCameraInput = document.getElementById('ipCameraInput');
    const cameraForm = document.getElementById('cameraForm');
    const videoFeed = document.getElementById('videoFeed');

    cameraSource.addEventListener('change', function() {
        if (this.value === 'ip') {
            ipCameraInput.classList.remove('d-none');
        } else {
            ipCameraInput.classList.add('d-none');
        }
    });

    cameraForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        let source = cameraSource.value;
        if (source === 'ip') {
            source = document.getElementById('ipUrl').value;
            if (!source) {
                alert('Please enter a valid IP camera URL');
                return;
            }
        }

        // Notify backend to restart camera
        fetch('/set_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: source })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                videoFeed.src = '/video_feed?' + new Date().getTime(); // Append cache-busting timestamp
            }
        })
        .catch(error => console.error('Error:', error));
    });
});