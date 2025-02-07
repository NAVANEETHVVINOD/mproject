document.addEventListener("DOMContentLoaded", () => {
    const API_URL = "http://localhost:8080/asr";

    const elements = {
        startBtn: document.getElementById("startRecord"),
        stopBtn: document.getElementById("stopRecord"),
        uploadBtn: document.getElementById("uploadButton"),
        status: document.getElementById("status"),
        results: document.getElementById("results"),
        spinner: document.getElementById("loading-spinner"),
        transcription: document.getElementById("transcription"),
        translation: document.getElementById("translation"),
        emrData: document.getElementById("emr-data"),
        suggestions: document.getElementById("suggestions"),
        downloadBtn: document.getElementById("downloadReport"),
    };

    let mediaRecorder;
    let audioChunks = [];
    let isRecorded = false;

    const initUI = () => {
        elements.stopBtn.disabled = true;
        elements.results.style.display = "none";
        elements.spinner.style.display = "none";
    };

    const checkRecordingSupport = () => {
        if (!navigator.mediaDevices?.getUserMedia) {
            setStatus("Audio recording not supported in this browser", "error");
            elements.startBtn.disabled = true;
            return false;
        }
        return true;
    };

    const processAudio = async (audioBlob, recorded = false) => {
        showLoading();
        try {
            const formData = new FormData();
            formData.append("audio", audioBlob);

            const headers = {
                'audio-source': recorded ? 'recorded' : 'uploaded'
            };

            const response = await fetch(API_URL, {
                method: "POST",
                body: formData,
                credentials: "include",
                headers: headers
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status} - ${await response.text()}`);
            }
            const data = await response.json();
            displayResults(data);
            setStatus("Processing complete!", "success");
        } catch (error) {
            setStatus(error.message, "error");
            console.error("Processing error:", error);
        } finally {
            hideLoading();
        }
    };

    const showLoading = () => {
        elements.spinner.style.display = "block";
        elements.results.style.display = "none";
    };

    const hideLoading = () => {
        elements.spinner.style.display = "none";
        elements.results.style.display = "block";
    };

    const setStatus = (message, type = "info") => {
        elements.status.textContent = message;
        elements.status.className = `alert alert-${type}`;
        elements.status.style.display = "block";

        if (type === "error") {
            setTimeout(() => {
                elements.status.style.display = "none";
            }, 5000);
        }
    };

    const displayResults = (data) => {
        elements.transcription.textContent = data.transcription || "N/A";
        elements.translation.textContent = data.translation || "N/A";
        displayJSON(data.emr, elements.emrData);
        displayJSON(data.suggestions, elements.suggestions);
        elements.results.style.display = "block";
    };

    const displayJSON = (data, container) => {
        if (!data || typeof data !== 'object') {
            container.innerHTML = '<div class="result-item">No data available</div>';
            return;
        }
        
        container.innerHTML = Object.entries(data)
            .map(([key, value]) => `<div class="result-item"><strong>${key}:</strong> <span>${value}</span></div>`)
            .join("");
    };

    const downloadReport = () => {
        try {
            const patientName = document.getElementById("patientNameInput").value || "Unknown";
            const patientAge = document.getElementById("patientAgeInput").value || "N/A";
            const patientAddress = document.getElementById("patientAddressInput").value || "N/A";
            const patientSex = document.getElementById("patientSexInput").value || "N/A";
            const patientPhone = document.getElementById("patientPhoneInput").value || "N/A";
            
            const currentDate = new Date().toLocaleDateString();
            const currentTime = new Date().toLocaleTimeString();

            const reportContent = `
=== MediTech Hospital - Medical Report ===
Date: ${currentDate}
Time: ${currentTime}

Patient Information:
-------------------
Name: ${patientName}
Age: ${patientAge}
Sex: ${patientSex}
Address: ${patientAddress}
Phone: ${patientPhone}

Transcription:
-------------
${elements.transcription.textContent}

Translation:
-----------
${elements.translation.textContent}

EMR Data:
---------
${elements.emrData.innerText}

Medical Suggestions:
------------------
${elements.suggestions.innerText}

=== End of Report ===
            `.trim();

            const blob = new Blob([reportContent], { type: "text/plain;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `medical_report_${patientName}_${currentDate.replace(/\//g, '-')}.txt`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Error generating report:", error);
            setStatus("Error generating report", "error");
        }
    };

    const setupEventListeners = () => {
        elements.startBtn.addEventListener("click", async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
                    await processAudio(audioBlob, true);
                };

                mediaRecorder.start();
                elements.startBtn.disabled = true;
                elements.stopBtn.disabled = false;
                setStatus("Recording...");
            } catch (error) {
                setStatus(`Recording error: ${error.message}`, "error");
                console.error("Recording error:", error);
            }
        });

        elements.stopBtn.addEventListener("click", () => {
            if (mediaRecorder?.state === "recording") {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                elements.startBtn.disabled = false;
                elements.stopBtn.disabled = true;
                setStatus("Processing...");
            }
        });

        elements.uploadBtn.addEventListener("click", () => {
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = "audio/*";
            fileInput.onchange = async (e) => {
                const file = e.target.files?.[0];
                if (file) await processAudio(file, false);
            };
            fileInput.click();
        });

        elements.downloadBtn.addEventListener("click", downloadReport);
    };

    const initializeApp = () => {
        initUI();
        if (checkRecordingSupport()) {
            setupEventListeners();
            setStatus("Ready to record");
        }
    };

    initializeApp();
});