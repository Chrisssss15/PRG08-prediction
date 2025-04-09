import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import testdata from './testdata.json' with {type: 'json'};



const enableWebcamButton = document.getElementById("webcamButton")
const logButton = document.getElementById("logButton")

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")
const statusDiv = document.getElementById("status")
const resultDiv = document.getElementById("result")


const drawUtils = new DrawingUtils(canvasCtx)
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;

let image = document.querySelector("#myimage")
let nn;


function createNeuralNetwork() {
    ml5.setBackend('webgl');
    nn = ml5.neuralNetwork({task: 'classification', debug: true});

    const option = {
        model: "model/model.json",
        metadata: "model/model_meta.json",
        weights: "model/model.weights.bin"

    }

    nn.load(option, createHandLandmarker);
}


/********************************************************************
// CREATE THE POSE DETECTOR
********************************************************************/
const createHandLandmarker = async () => {
    console.log("Loading model is loaded!");

    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
    console.log("model loaded, you can start webcam")
    
    enableWebcamButton.addEventListener("click", (e) => enableCam(e))
    // logButton.addEventListener("click", (e) => classifyHand(e)) 
    logButton.addEventListener("click", (e) => calculateLetter(e)) 

}

/********************************************************************
// START THE WEBCAM
********************************************************************/
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
// START PREDICTIONS    
********************************************************************/
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now())

    let hand = results.landmarks[0]
    if(hand) {
        let thumb = hand[4]
        image.style.transform = `translate(${video.videoWidth - thumb.x * video.videoWidth}px, ${thumb.y * video.videoHeight}px)`
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for(let hand of results.landmarks){
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
       window.requestAnimationFrame(predictWebcam)
    }
}

// function classifyHand() {
//     // console.log(results.landmarks[0]); // array van objecten met x,y,z
//     let numbersOnly = []
//     let hand = results.landmarks[0]
//     for(let point of hand){
//         numbersOnly.push(point.x, point.y, point.z)
//     }

//     // console.log(numbersOnly);
//     nn.classify(numbersOnly, (results) => {
//         // console.log(results);
//         console.log(`I am ${results[0].confidence.toFixed(2) * 100}% sure that this is a ${results[0].label}`);
//         statusDiv.textContent = `I think this pose is a ${results[0].label}. I am ${results[0].confidence.toFixed(2) * 100}% sure.`;
//     })

//     // statusDiv.textContent = `I think this pose is a ${results[0].label}. I am ${results[0].confidence.toFixed(2) * 100}% sure.`;
// }


async function calculateLetter() {
    let c = 0;
    let total = testdata.length;

    for (let pose of testdata) {
        const poseArray = pose.data;
        const result = await classifyAsync(poseArray);
        console.log(result.label, pose.label);
        if (result.label == pose.label) {
            c += 1 ;
        }
    }
    showInBrowser(total, c);

}

function showInBrowser(total, c) {
    const resultDiv = document.getElementById("result");
    resultDiv.textContent = `${c}/ ${total} are correct!, Accruacy is ${Math.round((c / total) * 100)}%`;
}

function classifyAsync(input) {
    return new Promise((resolve, reject) => {
        nn.classify(input, (result) => {
            if (result && result[0]) {
                resolve(result[0]);
            } else {
                reject(new Error("No result found"));
            }
        });
    });
}



/********************************************************************
// LOG HAND COORDINATES IN THE CONSOLE
********************************************************************/
function logAllHands(){
    for (let hand of results.landmarks) {
        // console.log(hand)
        console.log(hand[4])
    }
}

/********************************************************************
// START THE APP
********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createNeuralNetwork();
}