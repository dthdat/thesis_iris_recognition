(function () {
  "use strict";

  var config = window.IRIS_CONFIG || {};
  var defaultThreshold = config.defaultThreshold || "0.45";
  var defaultMargin = config.defaultMargin || "0.05";
  var CAMERA_FRAME_INTERVAL_MS = 60;
  var CAMERA_RETRY_INTERVAL_MS = 700;
  var CAMERA_QUALITY_INTERVAL_MS = 650;
  var state = {
    queryFile: null,
    leftFile: null,
    rightFile: null,
    busy: false,
    cameraLive: false,
    cameraFrameTimer: null,
    cameraQualityTimer: null,
    cameraQualityBusy: false,
    cameraErrors: 0,
    cameraAnalysis: null,
    cameraFps: 0,
    previewMode: "contrast",
    captureTarget: "query",
    stableStartedAt: null,
    captureInFlight: false,
    autoCaptureArmed: true,
    crop: null,
    cameraCaptures: {
      query: null,
      left: null,
      right: null
    }
  };

  function $(id) {
    return document.getElementById(id);
  }

  function text(id, value) {
    var el = $(id);
    if (el) {
      el.textContent = value;
    }
  }

  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function toast(message) {
    var box = $("toast");
    box.textContent = message;
    box.classList.add("show");
    window.setTimeout(function () {
      box.classList.remove("show");
    }, 3600);
  }

  function setLoading(message, show) {
    var loader = $("loader");
    text("loaderText", message || "Processing...");
    loader.classList.toggle("show", !!show);
    loader.setAttribute("aria-hidden", show ? "false" : "true");
  }

  function setBusy(on, message) {
    state.busy = !!on;
    var buttons = document.querySelectorAll("[data-busy-disable]");
    for (var i = 0; i < buttons.length; i += 1) {
      buttons[i].disabled = state.busy;
    }
    setLoading(message, state.busy);
  }

  function setDecision(kind, main, sub) {
    var el = $("decisionText");
    el.className = "decision-text " + kind;
    el.textContent = main;
    text("decisionSub", sub);
  }

  function setEnroll(kind, main, sub) {
    var el = $("enrollDecision");
    el.className = "decision-text " + kind;
    el.textContent = main;
    text("enrollSub", sub);
  }

  function setCamera(kind, main, sub) {
    var el = $("cameraDecision");
    if (!el) {
      return;
    }
    el.className = "decision-text " + kind;
    el.textContent = main;
    text("cameraSub", sub);
  }

  function cameraLog(message) {
    var log = $("cameraLog");
    if (!log) {
      return;
    }
    var now = new Date();
    var stamp = now.toLocaleTimeString ? now.toLocaleTimeString() : "";
    var line = "[" + stamp + "] " + message;
    if (log.textContent === "Camera events will appear here.") {
      log.textContent = line;
    } else {
      log.textContent = line + "\n" + log.textContent;
    }
  }

  function asScore(value) {
    if (value === null || value === undefined || value === "") {
      return "-";
    }
    var number = Number(value);
    if (!isFinite(number)) {
      return "-";
    }
    return number.toFixed(6);
  }

  function setThreshold(value) {
    var number = parseFloat(value);
    if (!isFinite(number)) {
      number = parseFloat(defaultThreshold);
    }
    if (number < 0.10) {
      number = 0.10;
    }
    if (number > 0.95) {
      number = 0.95;
    }
    var fixed = number.toFixed(2);
    $("thresholdRange").value = fixed;
    $("thresholdText").value = fixed;
    text("thresholdLabel", fixed);
  }

  function appendRecognitionOptions(fd, target) {
    var targetEye = "";
    if (target === "left" || target === "right") {
      targetEye = target;
    } else if ($("queryEye")) {
      targetEye = $("queryEye").value || "";
    }
    fd.append("query_eye", targetEye);
    fd.append("same_side_only", $("sameSideOnly") && $("sameSideOnly").checked ? "1" : "0");
    fd.append("margin_enabled", $("marginEnabled") && $("marginEnabled").checked ? "1" : "0");
    fd.append("margin", $("marginText") && $("marginText").value ? $("marginText").value : defaultMargin);
  }

  function recognitionSubtext(data) {
    if (data.matched) {
      return "Identity: " + data.name;
    }
    if (data.threshold_pass && data.margin_enabled && !data.margin_pass) {
      return "Rejected by margin check.";
    }
    return "Best score below threshold.";
  }

  function switchTab(tabId) {
    if (tabId !== "camera" && state.cameraLive) {
      stopCamera();
      releaseCameraDevice(false);
    }

    var tabs = document.querySelectorAll(".tab-button");
    var panels = document.querySelectorAll(".panel");
    var i;
    for (i = 0; i < tabs.length; i += 1) {
      tabs[i].classList.toggle("active", tabs[i].getAttribute("data-tab") === tabId);
    }
    for (i = 0; i < panels.length; i += 1) {
      panels[i].classList.toggle("active", panels[i].id === tabId);
    }
    if (tabId === "database") {
      loadUsers();
    }
  }

  function setupTabs() {
    var tabs = document.querySelectorAll(".tab-button");
    for (var i = 0; i < tabs.length; i += 1) {
      tabs[i].addEventListener("click", function () {
        switchTab(this.getAttribute("data-tab"));
      });
    }
  }

  function setupDrop(dropId, inputId, previewId, key, after) {
    var drop = $(dropId);
    var input = $(inputId);
    var preview = $(previewId);

    function handle(file) {
      if (!file) {
        return;
      }
      state[key] = file;
      preview.src = URL.createObjectURL(file);
      drop.classList.add("has-image");
      if (after) {
        after(file);
      }
    }

    drop.addEventListener("dragover", function (event) {
      event.preventDefault();
      drop.classList.add("drag");
    });

    drop.addEventListener("dragleave", function () {
      drop.classList.remove("drag");
    });

    drop.addEventListener("drop", function (event) {
      event.preventDefault();
      drop.classList.remove("drag");
      handle(event.dataTransfer.files[0]);
    });

    input.addEventListener("change", function () {
      handle(input.files[0]);
    });
  }

  function updateHealth() {
    return fetch("/api/health", { cache: "no-store" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        text("systemStatus", data.ok ? "Online" : "Error");
        text("userCount", data.db_count);
        text("homeUserCount", data.db_count);
        text("serverIp", String(data.ip || "Jetson") + ":" + String(data.port || config.serverPort || "8000"));
        if (data.camera) {
          text("homeCameraState", data.camera.preview_running ? "Streaming" : "Standby");
          text("homeCameraDetail", data.camera.last_error || (data.camera.preview_running ? "NoIR preview is connected." : "Open Live Capture to start the NoIR sensor."));
          text("homeFps", data.camera.measured_fps ? Number(data.camera.measured_fps).toFixed(1) + " FPS" : "—");
        }
        if (data.threshold !== undefined && !state.busy) {
          defaultThreshold = Number(data.threshold).toFixed(2);
        }
      })
      .catch(function () {
        text("systemStatus", "Offline");
      });
  }

  function loadUsers() {
    var grid = $("usersGrid");
    grid.innerHTML = '<div class="empty-state">Loading users...</div>';
    return fetch("/api/users", { cache: "no-store" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        text("userCount", data.count || 0);
        if (!data.users || !data.users.length) {
          grid.innerHTML = '<div class="empty-state">No enrolled users yet.</div>';
          return;
        }
        var html = "";
        for (var i = 0; i < data.users.length; i += 1) {
          var user = data.users[i];
          html += '<article class="user-card">';
          html += '<div class="user-name">' + esc(user.name) + '</div>';
          html += '<span class="badge ' + (user.left ? "" : "off") + '">Left ' + (user.left ? "OK" : "Missing") + '</span>';
          html += '<span class="badge ' + (user.right ? "" : "off") + '">Right ' + (user.right ? "OK" : "Missing") + '</span>';
          html += '<div class="user-date">Created: ' + esc(user.created_at || "unknown") + '</div>';
          html += '</article>';
        }
        grid.innerHTML = html;
      })
      .catch(function () {
        grid.innerHTML = '<div class="empty-state">Could not load database.</div>';
      });
  }

  function recognize() {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }
    if (!state.queryFile) {
      toast("Choose an iris image first.");
      return;
    }

    var fd = new FormData();
    fd.append("image", state.queryFile);
    fd.append("threshold", $("thresholdText").value || defaultThreshold || "0.45");
    appendRecognitionOptions(fd, null);

    setBusy(true, "Running recognition...");
    fetch("/api/recognize", { method: "POST", body: fd })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          setDecision("error", "ERROR", data.error || "Recognition failed");
          toast(data.error || "Recognition failed");
          return;
        }

        text("timeMetric", String(data.elapsed_ms) + " ms");
        text("scoreMetric", asScore(data.score));
        text("eyeMetric", data.eye || "-");
        text("thresholdLabel", Number(data.threshold).toFixed(2));

        if (data.matched) {
          text("identityMetric", data.name || "-");
          setDecision("match", "MATCH FOUND", recognitionSubtext(data));
        } else {
          text("identityMetric", "Unknown");
          setDecision("nomatch", "NO MATCH", recognitionSubtext(data));
        }

        if (data.top_scores && data.top_scores.length) {
          var rows = [];
          if (data.score_margin !== null && data.score_margin !== undefined) {
            rows.push("Margin: " + asScore(data.score_margin) + " / required " + asScore(data.required_margin));
            rows.push("Second different identity: " + asScore(data.second_best_different_identity_score));
            rows.push("");
          }
          for (var i = 0; i < data.top_scores.length; i += 1) {
            var score = data.top_scores[i];
            rows.push(
              String(i + 1) + ". " + score.user + " / " + score.eye + " : " + asScore(score.score)
            );
          }
          text("scoreLog", rows.join("\n"));
        } else {
          text("scoreLog", "No users in database.");
        }
      })
      .catch(function (error) {
        setDecision("error", "ERROR", String(error));
        toast("Request failed: " + String(error));
      })
      .then(function () {
        setBusy(false);
        updateHealth();
      });
  }

  function enroll() {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }

    var name = $("enrollName").value.replace(/^\s+|\s+$/g, "");
    if (!name) {
      toast("Enter user name.");
      return;
    }
    if (!state.leftFile || !state.rightFile) {
      toast("Choose both left and right eye images.");
      return;
    }

    var fd = new FormData();
    fd.append("name", name);
    fd.append("left", state.leftFile);
    fd.append("right", state.rightFile);

    setBusy(true, "Enrolling both eyes...");
    fetch("/api/register", { method: "POST", body: fd })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          setEnroll("error", "ERROR", data.error || "Enrollment failed");
          toast(data.error || "Enrollment failed");
          return;
        }
        setEnroll("match", "ENROLLED", "User added: " + data.name);
        text("leftTimeMetric", String(data.left_ms) + " ms");
        text("rightTimeMetric", String(data.right_ms) + " ms");
        text("enrollTimeMetric", String(data.elapsed_ms) + " ms");
        text("enrollDbMetric", String(data.db.count) + " users");
        toast("Registered both eyes for " + data.name);
        updateHealth();
        loadUsers();
      })
      .catch(function (error) {
        setEnroll("error", "ERROR", String(error));
        toast("Request failed: " + String(error));
      })
      .then(function () {
        setBusy(false);
      });
  }

  function clearRecognition() {
    state.queryFile = null;
    $("queryInput").value = "";
    $("queryPreview").src = "";
    $("queryDrop").classList.remove("has-image");
    text("identityMetric", "-");
    text("eyeMetric", "-");
    text("scoreMetric", "-");
    text("timeMetric", "-");
    text("scoreLog", "Top matches will appear here.");
    setDecision("idle", "READY", "Choose an image to begin.");
  }

  function clearEnrollment() {
    state.leftFile = null;
    state.rightFile = null;
    $("leftInput").value = "";
    $("rightInput").value = "";
    $("leftPreview").src = "";
    $("rightPreview").src = "";
    $("leftDrop").classList.remove("has-image");
    $("rightDrop").classList.remove("has-image");
    $("enrollName").value = "";
    text("leftTimeMetric", "-");
    text("rightTimeMetric", "-");
    text("enrollTimeMetric", "-");
    text("enrollDbMetric", "-");
    setEnroll("idle", "WAITING", "Enter a name and choose both eyes.");
  }

  function deleteAll() {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }
    if (!window.confirm("Delete all enrolled users?")) {
      return;
    }
    setBusy(true, "Clearing database...");
    fetch("/api/delete_all", { method: "POST" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        toast(data.message || "Database cleared");
        updateHealth();
        loadUsers();
      })
      .catch(function (error) {
        toast("Delete failed: " + String(error));
      })
      .then(function () {
        setBusy(false);
      });
  }

  function scheduleCameraFrame(delay) {
    if (state.cameraFrameTimer) {
      window.clearTimeout(state.cameraFrameTimer);
      state.cameraFrameTimer = null;
    }
    if (!state.cameraLive) {
      return;
    }
    state.cameraFrameTimer = window.setTimeout(function () {
      state.cameraFrameTimer = null;
      cameraFrameLoop();
    }, delay);
  }

  function scheduleCameraQuality(delay) {
    if (state.cameraQualityTimer) {
      window.clearTimeout(state.cameraQualityTimer);
      state.cameraQualityTimer = null;
    }
    if (!state.cameraLive) {
      return;
    }
    state.cameraQualityTimer = window.setTimeout(function () {
      state.cameraQualityTimer = null;
      cameraQualityLoop();
    }, delay);
  }

  function drawRotatedImage(canvas, image, rotation, filter) {
    var width = image.width || image.naturalWidth;
    var height = image.height || image.naturalHeight;
    var swap = rotation === 90 || rotation === 270;
    canvas.width = swap ? height : width;
    canvas.height = swap ? width : height;
    var ctx = canvas.getContext("2d");
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.filter = filter || "none";
    if (rotation === 90) {
      ctx.translate(canvas.width, 0);
      ctx.rotate(Math.PI / 2);
    } else if (rotation === 180) {
      ctx.translate(canvas.width, canvas.height);
      ctx.rotate(Math.PI);
    } else if (rotation === 270) {
      ctx.translate(0, canvas.height);
      ctx.rotate(-Math.PI / 2);
    }
    ctx.drawImage(image, 0, 0);
    ctx.restore();
  }

  function drawDetectionOverlay(canvas, analysis) {
    if (!analysis || !analysis.eye_box || state.previewMode === "crop") {
      return;
    }
    var box = analysis.eye_box;
    var ctx = canvas.getContext("2d");
    ctx.save();
    ctx.strokeStyle = analysis.ready ? "#2dd4bf" : "#fbbf24";
    ctx.lineWidth = Math.max(3, canvas.width / 260);
    ctx.setLineDash([Math.max(8, canvas.width / 70), Math.max(5, canvas.width / 110)]);
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    if (analysis.iris_circle) {
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(
        analysis.iris_circle.x,
        analysis.iris_circle.y,
        analysis.iris_circle.radius,
        0,
        Math.PI * 2
      );
      ctx.stroke();
    }
    ctx.restore();
  }

  function renderCameraFrame(image) {
    var rawCanvas = $("cameraRawCanvas");
    var processedCanvas = $("cameraProcessedCanvas");
    if (!rawCanvas || !processedCanvas) {
      return;
    }
    drawRotatedImage(rawCanvas, image, 0, "none");

    var analysis = state.cameraAnalysis;
    var rotationSetting = $("cameraRotation") ? $("cameraRotation").value : "auto";
    var rotation = rotationSetting === "auto" ? (analysis ? Number(analysis.rotation || 0) : 0) : Number(rotationSetting || 0);
    var filter = "none";
    if (state.previewMode === "grayscale") {
      filter = "grayscale(1) brightness(1.18)";
    } else if (state.previewMode === "contrast" || state.previewMode === "crop") {
      var measuredBrightness = analysis && analysis.metrics ? Number(analysis.metrics.brightness || 90) : 90;
      var adaptiveBrightness = Math.max(1.08, Math.min(1.75, 112 / Math.max(35, measuredBrightness)));
      filter = "grayscale(1) contrast(1.32) brightness(" + adaptiveBrightness.toFixed(2) + ")";
    }
    drawRotatedImage(processedCanvas, image, rotation, filter);

    if (state.previewMode === "crop" && analysis && analysis.crop_box) {
      var box = analysis.crop_box;
      var source = document.createElement("canvas");
      drawRotatedImage(source, image, rotation, filter);
      processedCanvas.width = Math.max(1, box.width);
      processedCanvas.height = Math.max(1, box.height);
      var cropContext = processedCanvas.getContext("2d");
      cropContext.drawImage(
        source,
        box.x,
        box.y,
        box.width,
        box.height,
        0,
        0,
        box.width,
        box.height
      );
    } else {
      drawDetectionOverlay(processedCanvas, analysis);
    }
  }

  function decodeCameraBlob(blob) {
    if (window.createImageBitmap) {
      return window.createImageBitmap(blob);
    }
    return new Promise(function (resolve, reject) {
      var image = new Image();
      var url = URL.createObjectURL(blob);
      image.onload = function () {
        URL.revokeObjectURL(url);
        resolve(image);
      };
      image.onerror = function () {
        URL.revokeObjectURL(url);
        reject(new Error("Could not decode camera JPEG"));
      };
      image.src = url;
    });
  }

  function cameraFrameLoop() {
    if (!state.cameraLive) {
      return;
    }
    if (document.hidden) {
      scheduleCameraFrame(1000);
      return;
    }
    var frame = $("cameraFrame");
    fetch("/api/camera/frame?ts=" + String(Date.now()), { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Camera frame returned " + String(response.status));
        }
        state.cameraFps = Number(response.headers.get("X-Camera-FPS") || state.cameraFps || 0);
        return response.blob();
      })
      .then(decodeCameraBlob)
      .then(function (image) {
        renderCameraFrame(image);
        if (image.close) {
          image.close();
        }
        state.cameraErrors = 0;
        if (frame) {
          frame.classList.add("live");
        }
        $("cameraConnectionDot").className = "online";
        text("cameraConnectionText", "Camera connected");
        text("homeCameraState", "Streaming");
        if (state.cameraLive) {
          scheduleCameraFrame(CAMERA_FRAME_INTERVAL_MS);
        }
      })
      .catch(function (error) {
        state.cameraErrors += 1;
        // Keep the last decoded canvas visible. A transient read error must not
        // replace it with the browser's broken-image placeholder.
        $("cameraConnectionDot").className = "reconnecting";
        text("cameraConnectionText", "Reconnecting");
        if (state.cameraErrors >= 3) {
          setCamera("error", "CAMERA ISSUE", "Keeping the last good frame while reconnecting.");
          cameraLog(String(error));
        }
        if (state.cameraLive) {
          scheduleCameraFrame(CAMERA_RETRY_INTERVAL_MS);
        }
      });
  }

  function startCamera() {
    if (state.cameraLive) {
      return;
    }
    state.cameraLive = true;
    $("cameraStartBtn").disabled = true;
    text("cameraStartBtn", "Camera running");
    state.autoCaptureArmed = true;
    state.stableStartedAt = null;
    setCamera("idle", "LIVE", "Follow the quality guidance for automatic capture.");
    $("cameraConnectionDot").className = "reconnecting";
    text("cameraConnectionText", "Starting camera");
    cameraLog("Live camera started.");
    scheduleCameraFrame(0);
    scheduleCameraQuality(300);
  }

  function stopCamera() {
    state.cameraLive = false;
    if (state.cameraFrameTimer) {
      window.clearTimeout(state.cameraFrameTimer);
      state.cameraFrameTimer = null;
    }
    if (state.cameraQualityTimer) {
      window.clearTimeout(state.cameraQualityTimer);
      state.cameraQualityTimer = null;
    }
    state.stableStartedAt = null;
    $("cameraStartBtn").disabled = false;
    text("cameraStartBtn", "Start camera");
    var frame = $("cameraFrame");
    if (frame) {
      frame.classList.remove("live");
    }
    $("cameraConnectionDot").className = "";
    text("cameraConnectionText", "Camera idle");
    text("homeCameraState", "Standby");
    setCamera("idle", "STOPPED", "Camera stream is stopped.");
    cameraLog("Live camera stopped.");
  }

  function releaseCameraDevice(showStatus) {
    return fetch("/api/camera/release", { method: "POST" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          if (showStatus) {
            setCamera("error", "ERROR", data.error || "Camera release failed");
          }
          return;
        }
        if (showStatus) {
          setCamera("idle", "RELEASED", "Camera device released.");
          cameraLog("Camera device released.");
        }
      })
      .catch(function (error) {
        if (showStatus) {
          setCamera("error", "ERROR", String(error));
        }
      });
  }

  function releaseCamera() {
    stopCamera();
    releaseCameraDevice(true);
  }

  function setMetric(dotId, valueId, value, status) {
    var dot = $(dotId);
    if (dot) {
      dot.className = "metric-dot " + (status || "");
    }
    text(valueId, value);
  }

  function metricStatus(pass, caution) {
    if (pass) {
      return "good";
    }
    return caution ? "warn" : "bad";
  }

  function stableDuration() {
    return Number($("stableDuration") && $("stableDuration").value || 1200);
  }

  function resetStableProgress(label) {
    state.stableStartedAt = null;
    $("stableProgress").style.width = "0%";
    text("stableText", label || "Waiting");
  }

  function updateQualityUi(data) {
    var analysis = data.analysis;
    if (!analysis) {
      text("qualityLabel", "Analyzing camera");
      text("qualityGuidance", "Detecting face, eye, and iris…");
      return;
    }
    var metrics = analysis.metrics || {};
    state.cameraAnalysis = analysis;
    var score = Number(analysis.score || 0);
    text("qualityScore", String(Math.round(score)));
    $("qualityRing").style.setProperty("--score", String(score * 3.6) + "deg");
    $("qualityRing").className = "score-ring " + (analysis.ready ? "ready" : analysis.state || "");
    text("qualityLabel", analysis.ready ? "Capture ready" : (analysis.state === "searching" ? "Looking for an eye" : "Adjust position"));
    text("qualityGuidance", analysis.guidance || "Follow the guide.");

    setMetric("focusDot", "focusValue", Math.round(Number(metrics.sharpness || 0)).toString(), metricStatus(Number(metrics.sharpness || 0) >= 18, Number(metrics.sharpness || 0) >= 10));
    setMetric("lightDot", "lightValue", Math.round(Number(metrics.brightness || 0)).toString(), metricStatus(Number(metrics.brightness || 0) >= 32 && Number(metrics.brightness || 0) <= 232, Number(metrics.brightness || 0) >= 20 && Number(metrics.brightness || 0) <= 242));
    setMetric("irisDot", "irisValue", Math.round(Number(metrics.iris_confidence || 0) * 100) + "%", metricStatus(Number(metrics.iris_confidence || 0) >= 0.22, Number(metrics.iris_confidence || 0) >= 0.12));
    setMetric("sizeDot", "sizeValue", Math.round(Number(metrics.eye_size || 0) * 100) + "%", metricStatus(Number(metrics.eye_size || 0) >= 0.075 && Number(metrics.eye_size || 0) <= 0.68, Number(metrics.eye_size || 0) >= 0.04));
    setMetric("motionDot", "motionValue", Math.round(Number(metrics.motion_stability || 0) * 100) + "%", metricStatus(Number(metrics.motion_stability || 0) >= 0.62, Number(metrics.motion_stability || 0) >= 0.40));
    var fps = data.camera && Number(data.camera.measured_fps || 0) || state.cameraFps || 0;
    state.cameraFps = fps;
    setMetric("fpsDot", "cameraFpsValue", fps ? fps.toFixed(1) + " FPS" : "Starting", metricStatus(fps >= 6, fps >= 2));
    text("homeFps", fps ? fps.toFixed(1) + " FPS" : "—");
    text("homeCameraDetail", analysis.guidance || "NoIR preview is connected.");

    if (analysis.ready && state.autoCaptureArmed && !state.captureInFlight) {
      if (state.stableStartedAt === null) {
        state.stableStartedAt = Date.now();
      }
      var elapsed = Date.now() - state.stableStartedAt;
      var progress = Math.min(100, Math.round(100 * elapsed / stableDuration()));
      $("stableProgress").style.width = String(progress) + "%";
      text("stableText", progress < 100 ? "Hold still " + String(progress) + "%" : "Ready");
      if (progress >= 100 && $("autoCaptureEnabled").checked) {
        captureGuided();
      }
    } else if (!state.captureInFlight) {
      resetStableProgress(analysis.ready && !state.autoCaptureArmed ? "Captured" : "Waiting for quality");
    }
  }

  function cameraQualityLoop() {
    if (!state.cameraLive) {
      return;
    }
    if (state.cameraQualityBusy || document.hidden) {
      scheduleCameraQuality(CAMERA_QUALITY_INTERVAL_MS);
      return;
    }
    state.cameraQualityBusy = true;
    var rotation = $("cameraRotation") ? $("cameraRotation").value : "auto";
    var threshold = $("qualityThreshold") ? $("qualityThreshold").value : "72";
    fetch(
      "/api/camera/quality?rotation=" + encodeURIComponent(rotation) +
      "&threshold=" + encodeURIComponent(threshold) +
      "&ts=" + String(Date.now()),
      { cache: "no-store" }
    )
      .then(function (response) {
        return response.json().then(function (data) {
          if (!response.ok || !data.ok) {
            throw new Error(data.error || "Quality analysis failed");
          }
          return data;
        });
      })
      .then(updateQualityUi)
      .catch(function (error) {
        resetStableProgress("Analysis reconnecting");
        if (state.cameraErrors >= 3) {
          text("qualityGuidance", String(error));
        }
      })
      .then(function () {
        state.cameraQualityBusy = false;
        if (state.cameraLive) {
          scheduleCameraQuality(CAMERA_QUALITY_INTERVAL_MS);
        }
      });
  }

  function setCaptureTarget(target) {
    state.captureTarget = target;
    state.autoCaptureArmed = true;
    resetStableProgress("Waiting for quality");
    var buttons = document.querySelectorAll("[data-capture-target]");
    for (var i = 0; i < buttons.length; i += 1) {
      buttons[i].classList.toggle("active", buttons[i].getAttribute("data-capture-target") === target);
    }
    if (target === "query") {
      setCamera("idle", "RECOGNITION", "A quality capture will be recognized automatically.");
    } else {
      setCamera("idle", target === "left" ? "LEFT EYE" : "RIGHT EYE", "Hold this eye in the guide until it is saved.");
    }
  }

  function setPreviewMode(mode) {
    state.previewMode = mode;
    var buttons = document.querySelectorAll("[data-preview-mode]");
    for (var i = 0; i < buttons.length; i += 1) {
      buttons[i].classList.toggle("active", buttons[i].getAttribute("data-preview-mode") === mode);
    }
    var labels = { raw: "Raw", grayscale: "Grayscale", contrast: "NIR contrast", crop: "Eye crop" };
    text("processedModeLabel", labels[mode] || "Preview");
  }

  function captureGuided() {
    if (!state.cameraLive) {
      toast("Start the camera before capturing.");
      return;
    }
    if (state.captureInFlight) {
      return;
    }
    state.captureInFlight = true;
    resetStableProgress("Capturing…");
    setCamera("idle", "CAPTURING", "Running the final iris quality check.");
    var target = state.captureTarget;
    var fd = new FormData();
    fd.append("target", target);
    fd.append("rotation", $("cameraRotation") ? $("cameraRotation").value : "auto");
    fd.append("threshold", $("qualityThreshold") ? $("qualityThreshold").value : "72");
    fetch("/api/camera/auto-capture", { method: "POST", body: fd })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          throw new Error(data.error || "Capture failed");
        }
        if (!data.captured) {
          state.autoCaptureArmed = true;
          setCamera("nomatch", "TRY AGAIN", data.message || "Capture quality changed.");
          cameraLog(data.message || "Capture was rejected by the final quality check.");
          return;
        }
        state.autoCaptureArmed = false;
        updateCameraPreview(target, data.processed_image_url || data.image_url);
        cameraLog("Quality capture saved for " + target + " at score " + String(Math.round(data.analysis.score)) + ".");
        if (target === "query") {
          setCamera("match", "CAPTURED", "Recognition is starting automatically.");
          window.setTimeout(cameraRecognize, 180);
        } else if (target === "left") {
          setCamera("match", "LEFT SAVED", "Now capture the right eye.");
          window.setTimeout(function () { setCaptureTarget("right"); }, 700);
        } else {
          setCamera("match", "BOTH EYES READY", "Enter a name, then register this person.");
        }
      })
      .catch(function (error) {
        state.autoCaptureArmed = true;
        setCamera("error", "CAPTURE ERROR", String(error));
        cameraLog(String(error));
      })
      .then(function () {
        state.captureInFlight = false;
      });
  }

  function previewForTarget(target) {
    if (target === "left") {
      return "cameraLeftPreview";
    }
    if (target === "right") {
      return "cameraRightPreview";
    }
    return "cameraQueryPreview";
  }

  function updateCameraPreview(target, url) {
    state.cameraCaptures[target] = url;
    var img = $(previewForTarget(target));
    if (img) {
      img.src = url + "?ts=" + String(Date.now());
      if (img.parentElement) {
        img.parentElement.classList.add("has-capture");
      }
    }
  }

  function cropToolRect() {
    var tool = $("cameraPhotoTool");
    var img = $("cameraPhoto");
    if (!tool || !img || !img.naturalWidth) {
      return null;
    }
    return {
      tool: tool.getBoundingClientRect(),
      img: img.getBoundingClientRect(),
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight
    };
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function drawCrop(crop) {
    var rect = cropToolRect();
    var box = $("cropBox");
    if (!rect || !crop || crop.width < 4 || crop.height < 4) {
      state.crop = null;
      $("cameraPhotoTool").classList.remove("has-crop");
      return;
    }
    state.crop = crop;
    box.style.left = String(rect.img.left - rect.tool.left + crop.x) + "px";
    box.style.top = String(rect.img.top - rect.tool.top + crop.y) + "px";
    box.style.width = String(crop.width) + "px";
    box.style.height = String(crop.height) + "px";
    $("cameraPhotoTool").classList.add("has-crop");
  }

  function defaultCrop() {
    var rect = cropToolRect();
    if (!rect) {
      return;
    }
    var size = Math.min(rect.img.width, rect.img.height) * 0.32;
    drawCrop({
      x: (rect.img.width - size) / 2,
      y: (rect.img.height - size) / 2,
      width: size,
      height: size
    });
  }

  function cropToNatural() {
    var rect = cropToolRect();
    if (!rect || !state.crop) {
      return null;
    }
    var sx = rect.naturalWidth / rect.img.width;
    var sy = rect.naturalHeight / rect.img.height;
    return {
      x: Math.round(state.crop.x * sx),
      y: Math.round(state.crop.y * sy),
      width: Math.round(state.crop.width * sx),
      height: Math.round(state.crop.height * sy)
    };
  }

  function setupCropTool() {
    var tool = $("cameraPhotoTool");
    var active = false;
    var start = null;

    function point(event) {
      var rect = cropToolRect();
      if (!rect) {
        return null;
      }
      return {
        x: clamp(event.clientX - rect.img.left, 0, rect.img.width),
        y: clamp(event.clientY - rect.img.top, 0, rect.img.height)
      };
    }

    tool.addEventListener("pointerdown", function (event) {
      if (!$("cameraPhoto").naturalWidth) {
        return;
      }
      event.preventDefault();
      active = true;
      start = point(event);
      if (tool.setPointerCapture) {
        tool.setPointerCapture(event.pointerId);
      }
      drawCrop({ x: start.x, y: start.y, width: 1, height: 1 });
    });

    tool.addEventListener("pointermove", function (event) {
      if (!active || !start) {
        return;
      }
      event.preventDefault();
      var current = point(event);
      if (!current) {
        return;
      }
      drawCrop({
        x: Math.min(start.x, current.x),
        y: Math.min(start.y, current.y),
        width: Math.abs(current.x - start.x),
        height: Math.abs(current.y - start.y)
      });
    });

    tool.addEventListener("pointerup", function (event) {
      active = false;
      start = null;
      if (tool.releasePointerCapture) {
        try {
          tool.releasePointerCapture(event.pointerId);
        } catch (e) {}
      }
    });
  }

  function takeCameraPhoto() {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }
    setBusy(true, "Taking full-resolution photo...");
    fetch("/api/camera/photo", { method: "POST" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          setCamera("error", "ERROR", data.error || "Photo capture failed");
          cameraLog(data.error || "Photo capture failed");
          return;
        }
        var img = $("cameraPhoto");
        img.onload = function () {
          $("cameraPhotoTool").classList.add("has-photo");
          defaultCrop();
        };
        img.src = data.image_url + "?ts=" + String(Date.now());
        setCamera("idle", "PHOTO READY", "Draw a box around the iris.");
        cameraLog("Photo captured at " + String(data.width) + "x" + String(data.height) + ".");
      })
      .catch(function (error) {
        setCamera("error", "ERROR", String(error));
        cameraLog("Photo request failed: " + String(error));
      })
      .then(function () {
        setBusy(false);
      });
  }

  function submitCrop(purpose) {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }
    var crop = cropToNatural();
    if (!crop) {
      toast("Take a photo and draw a crop box first.");
      return;
    }
    var fd = new FormData();
    fd.append("purpose", purpose);
    fd.append("target", purpose === "query" ? "query" : state.captureTarget);
    fd.append("x", crop.x);
    fd.append("y", crop.y);
    fd.append("width", crop.width);
    fd.append("height", crop.height);

    setBusy(true, "Checking iris crop...");
    fetch("/api/camera/crop", { method: "POST", body: fd })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          setCamera("error", "ERROR", data.error || "Crop failed");
          cameraLog(data.error || "Crop failed");
          return;
        }
        if (!data.ready) {
          setCamera("nomatch", "TRY AGAIN", data.message || "Iris crop was not accepted.");
          cameraLog(data.message || "Crop was not accepted.");
          return;
        }
        updateCameraPreview(data.target, data.image_url);
        if (purpose === "query") {
          setCamera("match", "QUERY READY", "Recognize the cropped iris when ready.");
          cameraLog("Query crop ready. Auto side: " + data.auto_eye + ".");
        } else {
          setCamera("match", "EYE SAVED", "Auto slot: " + data.target + ".");
          cameraLog("Enrollment crop saved to " + data.target + " slot.");
        }
      })
      .catch(function (error) {
        setCamera("error", "ERROR", String(error));
        cameraLog("Crop request failed: " + String(error));
      })
      .then(function () {
        setBusy(false);
      });
  }

  function cameraRecognize() {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }
    var target = "query";
    var fd = new FormData();
    fd.append("target", target);
    fd.append("threshold", $("thresholdText").value || defaultThreshold || "0.45");
    appendRecognitionOptions(fd, target);

    setBusy(true, "Recognizing camera capture...");
    fetch("/api/camera/recognize", { method: "POST", body: fd })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          setCamera("error", "ERROR", data.error || "Camera recognition failed");
          cameraLog(data.error || "Camera recognition failed");
          return;
        }
        if (data.matched) {
          setCamera("match", "MATCH FOUND", "Identity: " + data.name + " / score " + asScore(data.score));
        } else {
          setCamera("nomatch", "NO MATCH", recognitionSubtext(data) + " Score " + asScore(data.score));
        }
        cameraLog("Recognition score " + asScore(data.score) + ", threshold " + asScore(data.threshold));
        text("identityMetric", data.name || (data.matched ? "-" : "Unknown"));
        text("eyeMetric", data.eye || "-");
        text("scoreMetric", asScore(data.score));
        text("timeMetric", String(data.elapsed_ms) + " ms");
        updateHealth();
      })
      .catch(function (error) {
        setCamera("error", "ERROR", String(error));
      })
      .then(function () {
        setBusy(false);
      });
  }

  function cameraEnroll() {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }
    var name = $("cameraEnrollName").value.replace(/^\s+|\s+$/g, "");
    if (!name) {
      toast("Enter a user name for camera enrollment.");
      return;
    }
    var fd = new FormData();
    fd.append("name", name);

    setBusy(true, "Enrolling camera captures...");
    fetch("/api/camera/register", { method: "POST", body: fd })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          setCamera("error", "ERROR", data.error || "Camera enrollment failed");
          cameraLog(data.error || "Camera enrollment failed");
          return;
        }
        setCamera("match", "ENROLLED", "Camera captures enrolled for " + data.name);
        cameraLog("Enrolled " + data.name + " from camera captures.");
        toast("Registered camera captures for " + data.name);
        updateHealth();
        loadUsers();
      })
      .catch(function (error) {
        setCamera("error", "ERROR", String(error));
      })
      .then(function () {
        setBusy(false);
      });
  }

  function init() {
    setupTabs();
    setupDrop("queryDrop", "queryInput", "queryPreview", "queryFile", function () {
      setDecision("idle", "LOADED", "Preview ready.");
      if ($("autoRecognize").checked) {
        recognize();
      }
    });
    setupDrop("leftDrop", "leftInput", "leftPreview", "leftFile", function () {
      setEnroll("idle", "LEFT READY", "Left eye loaded.");
    });
    setupDrop("rightDrop", "rightInput", "rightPreview", "rightFile", function () {
      setEnroll("idle", "RIGHT READY", "Right eye loaded.");
    });

    $("thresholdRange").addEventListener("input", function () {
      setThreshold(this.value);
    });
    $("thresholdText").addEventListener("change", function () {
      setThreshold(this.value);
    });
    $("marginText").addEventListener("change", function () {
      var number = parseFloat(this.value);
      if (!isFinite(number)) {
        number = parseFloat(defaultMargin);
      }
      if (number < 0) {
        number = 0;
      }
      if (number > 0.50) {
        number = 0.50;
      }
      this.value = number.toFixed(2);
    });

    var presets = document.querySelectorAll("[data-threshold-preset]");
    for (var i = 0; i < presets.length; i += 1) {
      presets[i].addEventListener("click", function () {
        setThreshold(this.getAttribute("data-threshold-preset"));
      });
    }

    $("recognizeBtn").addEventListener("click", recognize);
    $("clearQueryBtn").addEventListener("click", clearRecognition);
    $("registerBtn").addEventListener("click", enroll);
    $("clearEnrollBtn").addEventListener("click", clearEnrollment);
    $("refreshDbBtn").addEventListener("click", loadUsers);
    $("deleteDbBtn").addEventListener("click", deleteAll);
    $("cameraStartBtn").addEventListener("click", startCamera);
    $("cameraCaptureNowBtn").addEventListener("click", captureGuided);
    $("cameraPhotoBtn").addEventListener("click", takeCameraPhoto);
    $("cameraStopBtn").addEventListener("click", stopCamera);
    $("cameraReleaseBtn").addEventListener("click", releaseCamera);
    $("cropQueryBtn").addEventListener("click", function () {
      submitCrop("query");
    });
    $("cropEnrollBtn").addEventListener("click", function () {
      submitCrop("enroll");
    });
    $("cameraRecognizeBtn").addEventListener("click", cameraRecognize);
    $("cameraEnrollBtn").addEventListener("click", cameraEnroll);

    var quickLinks = document.querySelectorAll("[data-open-tab]");
    for (var q = 0; q < quickLinks.length; q += 1) {
      quickLinks[q].addEventListener("click", function () {
        switchTab(this.getAttribute("data-open-tab"));
      });
    }
    var captureTargets = document.querySelectorAll("[data-capture-target]");
    for (var c = 0; c < captureTargets.length; c += 1) {
      captureTargets[c].addEventListener("click", function () {
        setCaptureTarget(this.getAttribute("data-capture-target"));
      });
    }
    var previewModes = document.querySelectorAll("[data-preview-mode]");
    for (var p = 0; p < previewModes.length; p += 1) {
      previewModes[p].addEventListener("click", function () {
        setPreviewMode(this.getAttribute("data-preview-mode"));
      });
    }
    $("qualityThreshold").addEventListener("input", function () {
      text("qualityThresholdValue", this.value);
      state.stableStartedAt = null;
    });
    $("cameraRotation").addEventListener("change", function () {
      state.cameraAnalysis = null;
      state.stableStartedAt = null;
    });

    document.addEventListener("visibilitychange", function () {
      if (!state.cameraLive) {
        return;
      }
      if (document.hidden) {
        if (state.cameraFrameTimer) {
          window.clearTimeout(state.cameraFrameTimer);
          state.cameraFrameTimer = null;
        }
        if (state.cameraQualityTimer) {
          window.clearTimeout(state.cameraQualityTimer);
          state.cameraQualityTimer = null;
        }
      } else {
        scheduleCameraFrame(0);
        scheduleCameraQuality(100);
      }
    });

    window.addEventListener("beforeunload", function () {
      stopCamera();
      if (navigator.sendBeacon) {
        navigator.sendBeacon("/api/camera/release", new Blob([], { type: "application/x-www-form-urlencoded" }));
      } else {
        fetch("/api/camera/release", { method: "POST", keepalive: true });
      }
    });

    setupCropTool();
    setCaptureTarget("query");
    setPreviewMode("contrast");
    setThreshold(defaultThreshold);
    $("marginText").value = Number(defaultMargin).toFixed(2);
    updateHealth();
    loadUsers();
    window.setInterval(updateHealth, 7000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
