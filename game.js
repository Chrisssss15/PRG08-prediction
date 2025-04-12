import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
// import testdata from './testdata.json' with {type: 'json'};

const quizQuestions = [
    {
        question: "Wat is 5 + 3?",
        options: ["6", "7", "8", "9"],
        correct: "c" // 8
    },
    {
        question: "Wat is 10 - 4?",
        options: ["6", "5", "4", "7"],
        correct: "a" // 6
    },
    {
        question: "Wat is 3 x 3?",
        options: ["6", "8", "9", "12"],
        correct: "c" // 9
    },
    {
        question: "Wat is 12 / 3?",
        options: ["3", "4", "6", "2"],
        correct: "b" // 4
    },
    {
        question: "Wat is 7 + 2?",
        options: ["8", "9", "10", "11"],
        correct: "b" // 9
    },
];

const handToAnswerMap = {
    "flex": "A",
    "duimpiee": "B",
    "love": "C",
    "boks": "D"
};

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

let currentQuestionIndex = 0;
let score = 0;
// const answer = handToAnswerMap[label];

let autoAnswerInterval;
let hasAnswered = false;



function showQuestion() {
    const q = quizQuestions[currentQuestionIndex];
    resultDiv.innerHTML = `
        <h2>${q.question}</h2>
        <ul>
            <li>a) 🤙🏼 ${q.options[0]}</li>
            <li>b) 👍 ${q.options[1]}</li>
            <li>c) 🫰🏼 ${q.options[2]}</li>
            <li>d) 👊🏼 ${q.options[3]}</li>
        </ul>
        <p>Maak een handgebaar om te antwoorden...</p>
    `;
}

// Start de eerste vraag
showQuestion();


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

const createHandLandmarker = async () => {
    console.log("Loading model is loaded!");

    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
    console.log("model loaded, you can start webcam")
    
    enableCam();
    logButton.addEventListener("click", (e) => classifyHand(e)) 

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
        if (!autoAnswerInterval && results.landmarks[0]) {
            autoAnswerInterval = setInterval(() => {
                console.log("autoAnswerInterval")
                if (!hasAnswered && results.landmarks[0]) {
                    classifyHand();
                    hasAnswered = true;
                }
            }, 5000); // Elke 2 sec checkt die of je een handgebaar maakt
        }
        window.requestAnimationFrame(predictWebcam);
    }

    

}


function classifyHand() {
    let numbersOnly = [];
    let hand = results.landmarks[0];
    for (let point of hand) {
        numbersOnly.push(point.x, point.y, point.z);
    }

    nn.classify(numbersOnly, (results) => {
        const label = results[0].label;
        const answer = handToAnswerMap[label];
        const q = quizQuestions[currentQuestionIndex];

        if (!answer) {
            statusDiv.textContent = `Onbekend gebaar: ${label}`;
            return;
        }

        const correct = q.correct === answer;
        if (correct) score++;
        
        statusDiv.textContent = correct
            ? `✅ Goed! ${answer} is het juiste antwoord.`
            : `❌ Fout. Jij deed ${answer}, maar het juiste antwoord was ${q.correct}`;
        
        document.getElementById("score").textContent = `Score: ${score}`;
        currentQuestionIndex++;

        if (currentQuestionIndex < quizQuestions.length) {
            setTimeout(() => {
                showQuestion();
                statusDiv.textContent = "";
            }, 2000); // wacht 2 sec
        } else {
            resultDiv.innerHTML = `
            <h2>🎉 Quiz afgerond!</h2>
            <p>Je score: ${score} van de ${quizQuestions.length}</p>
        `;            clearInterval(autoAnswerInterval);
        }
        setTimeout(() => {
            hasAnswered = false; // laat toe dat er weer opnieuw geantwoord wordt
        }, 5000); 
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