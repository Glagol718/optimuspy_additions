function copy_token() {
    // Get button by id
    var copyText = document.getElementById("token");
    // Copy button's text to clipboard
    navigator.clipboard.writeText(copyText.textContent);
    // Alert the copied text
    // alert("Copied the text: " + copyText.textContent);
}
