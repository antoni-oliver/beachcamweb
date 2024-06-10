/* 
* Image uploader js
*/

const image_query = "[name^=image]";
const canvas_query = ".canvas-img";
const form_query = "form.image-uploader";

/**
 * Initializa ImageUploaderForms events
 */
export function initImageUploaderForms() {
    hideAllSubmits();
    initSubmitEvent();
    initCanvasDrawEvent();
}

/* 
* Assures all submits are hidden at the start
*/
function hideAllSubmits() {
    $(`${form_query} :submit`).each(function () {
        $(this).hide();
    });
}

/* 
* Initialize Submit Event delegation
*/
function initSubmitEvent() {
    //event delegation for formSubmit
    $(document).on('click', `${form_query} :submit`, function (e) {
        e.preventDefault();
        let form = $(this).closest(form_query)[0];
        submitForm(form);
    });
}

function submitForm(form) {
    //TODO implement ajax
    alert("PALLA QUE VOY")
}

/**
 * Initialize Canvas Image drawing event delegation
 */
function initCanvasDrawEvent() {
    //event delegation for canvas viewer
    $(document).on('change', `${form_query} ${image_query}`, function (e) {
        //check files and if has canvas
        let form = $(e.target).closest(form_query)
        let canvas = form.find(canvas_query).first();
        let submit = form.find(':submit');
        if (e.target.files.length <= 0 || !canvas) {
            submit?.hide();
            return;
        }
        submit?.show();
        handleCanvasViewer(e.target.files[0], canvas[0]);
    });
}

function handleCanvasViewer(image, canvas) {
    try {
        const reader = new FileReader();
        reader.onload = function (e) {
            const ctx = canvas.getContext("2d");
            const img = new Image();
            img.src = e.target.result;
            img.onload = function () {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                canvas.style.display = "block";
            };
        };
        reader.readAsDataURL(image);
    } catch (error) {
        console.error(error);
    }
}

