var level = 1;
var left = document.getElementById("left");
var right = document.getElementById("right");
var secret = document.getElementById("secret_to_send");
var cover = document.getElementById("cover");
var summary = document.getElementById("summary");

pages = {
    1: "secret_to_send", 
    2: "cover",
    3: "hidden"
}


function updateButtons() {
    if(level = 1) {
        left.disabled = true;
    }
    else {
        left.disabled = false;
    }
}

updateButtons();

right.addEventListener("click", function() {
    

    switch (level) {

        case 1 :  
        secret.classList.add("make_none");
        cover.classList.remove("make_none");
        left.disabled = false;
        break;

        case 2 :
        summary.classList.remove("make_none");
        cover.classList.add("make_none");
        right.disabled = true;
        break
        
    }
    
    level += 1;
});


left.addEventListener("click", function() {
    switch(level) {
        case 2:
        secret.classList.remove("make_none");
        cover.classList.add("make_none");
        left.disabled = true;
        break

        case 3:
        cover.classList.remove("make_none");
        summary.classList.add("make_none");
        right.disabled = false;

    }

    level -= 1;
});