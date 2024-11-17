$(window).ready(function(){
    $(".boton").wrapInner('<div class=botontext></div>');
        
        $(".botontext").clone().appendTo( $(".boton") );
        
        $(".boton").append('<span class="twist"></span><span class="twist"></span><span class="twist"></span><span class="twist"></span>');
        
        $(".twist").css("width", "25%").css("width", "+=3px");
    });

    async function setupCamera() {
        const video = document.getElementById('video');
        video.src = "{{ url_for('video_feed') }}"; // Flask video feed URL
        video.width = 640;
        video.height = 480;
    
        return new Promise((resolve) => {
            video.onloadedmetadata = () => resolve(video);
        });
    }