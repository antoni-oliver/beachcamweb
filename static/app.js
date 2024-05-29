/**
 * JS file that includes home functionalities
 */

const TAG_IMG = "IMG";
let modalImageContainer;

//DOM has loaded
document.addEventListener("DOMContentLoaded", function () {
    modalImageContainer = document.getElementById('modalImage');
    showImageHandler()
});

/**
 * When a img is clicked, it sets the src image of the modal
 * to the clicked img
 */
function showImageHandler() {
    let imgContainer = document.getElementById('img-prediction-section');
    imgContainer.addEventListener('click', function (event) {
        let element = event.target;
        if (element.tagName === TAG_IMG) {
            modalImageContainer.src = element.src;
            modalImageContainer.parentElement.style = `background-image: url(${element.src})`;
        }
    });
}

/**
 * zooms over an element
 * @param {*} e event target
 */
function zoom(e) {
    var zoomer = e.currentTarget;
    e.offsetX ? offsetX = e.offsetX : offsetX = e.touches[0].pageX
    e.offsetY ? offsetY = e.offsetY : offsetX = e.touches[0].pageX
    x = offsetX / zoomer.offsetWidth * 100
    y = offsetY / zoomer.offsetHeight * 100
    zoomer.style.backgroundPosition = x + '% ' + y + '%';
}