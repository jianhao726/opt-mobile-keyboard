(() => {
  "use strict";
  function init() {
    const el = id => document.getElementById(id);
    const input = el("draft");
    const profiles = Object.fromEntries(["Solo", "User A", "User B"].map(name =>
      [name, { draft: "", cache: new Map(), trusted: new Set(), dismissed: new Set() }]));
    const logs = { solo: [], chat: [] };
    const prompts = {
      solo: [["Low battery", "My phone battery is at 1%, do you have a "],
        ["Train meetup", "Sure, I will meet you at the train "],
        ["Grocery run", "I am heading to the groc"], ["Lunch plans", "Hey! Are you free for lun"]],
      chat: [["Lunch invite", "Hey! Are you free for lunch today?", ""],
        ["Phone battery", "My phone battery is at 1%, do you have a charger?", "Yes, I have a "],
        ["Train meetup", "I just arrived at the station, where are you?", "I am on my "]]
    };
    let mode = "solo", version = 0, timer, busy = false, queued = false, preview = null;
    const profileName = () => mode === "solo" ? "Solo" : el("speaker").value;
    const profile = () => profiles[profileName()];
    const signature = (word, start) => JSON.stringify([word.toLowerCase(), start]);
    const make = (tag, cls, text) => {
      const node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text !== undefined) node.textContent = text;
      return node;
    };
    const button = (cls, text, action) => {
      const node = make("button", cls, text);
      node.type = "button";
      node.addEventListener("click", action);
      return node;
    };
    function focus(offset = input.value.length) {
      input.focus({preventScroll: true});
      input.setSelectionRange(offset, offset);
    }
    function status(text, state) {
      el("engine-status").textContent = text;
      el("engine-status").dataset.state = state;
    }
    function renderPreview() {
      let suffix = "";
      if (preview && input.value && input.selectionStart === input.value.length && input.selectionEnd === input.value.length) {
        const partial = input.value.match(/\S+$/u)?.[0] || "";
        if (preview.mode === "Next-Word") suffix = preview.word;
        else if (partial && preview.word.toLowerCase().startsWith(partial.toLowerCase())) suffix = preview.word.slice(partial.length);
      }
      el("draft-preview").hidden = el("preview-hint").hidden = !suffix;
      el("preview-prefix").textContent = suffix ? input.value : "";
      el("preview-tail").textContent = suffix;
      el("draft-preview").scrollLeft = input.scrollLeft;
    }
    function setDraft(text, caret) {
      input.value = text;
      profile().draft = text;
      if (!text) profile().dismissed.clear();
      el("send").disabled = !text.trim();
      if (caret !== undefined) focus(caret);
      schedule();
    }
    function learn(text, confirmed = false) {
      for (const token of text.split(/\s+/u)) {
        const word = token.replace(/^[.,!?;:'"]+|[.,!?;:'"]+$/g, "").toLowerCase();
        if (!/^\p{L}{2,64}$/u.test(word)) continue;
        if (profile().cache.has(word) || profile().cache.size < 1024)
          profile().cache.set(word, Math.min(1_000_000, (profile().cache.get(word) || 0) + 1));
        if (confirmed && profile().trusted.size < 1024) profile().trusted.add(word);
      }
      renderVocabulary();
    }
    function renderVocabulary() {
      el("profile-label").textContent = profileName();
      el("reset-cache").disabled = !profile().cache.size && !profile().trusted.size;
      const rows = [...profile().cache].sort((a,b) => b[1]-a[1]).map(([word,count]) => {
        const row = make("li", "vocab-item");
        row.append(make("span", "vocab-word", word), make("span", "vocab-count", count));
        return row;
      });
      el("vocabulary").replaceChildren(...(rows.length ? rows : [make("li", "", "No words learned in this session.")]));
    }
    function renderHistory() {
      const messages = logs[mode];
      el("history-count").textContent = `${messages.length} ${messages.length === 1 ? "message" : "messages"}`;
      el("history-empty").hidden = messages.length > 0;
      el("history").replaceChildren(...messages.map(({sender,text}) => {
        const row = make("div", "history-item"); row.dataset.speaker = sender;
        row.append(make("span", "history-sender", sender === "Solo" ? "You" : sender), make("p", "history-text", text));
        return row;
      }));
      el("history").scrollTop = el("history").scrollHeight;
    }
    function send() {
      const text = input.value.trim();
      if (!text) return;
      const sender = profileName();
      logs[mode].push({sender,text});
      learn(text); // Sending text changes frequency, but never confirms its spelling.
      profile().draft = ""; profile().dismissed.clear();
      if (mode === "chat" && el("auto-switch").checked) el("speaker").value = sender === "User A" ? "User B" : "User A";
      setDraft(profile().draft, profile().draft.length);
      el("send-status").textContent = `Sent${mode === "chat" ? " as " + sender : ""}.`;
      renderHistory(); renderVocabulary();
    }
    function choose(word, predictionMode, punctuation) {
      const current = input.value;
      const text = punctuation ? current.trimEnd() + word + " " : predictionMode === "Prefix-Completion"
        ? current.replace(/[^\s]*$/u, () => word + " ") : current + (current && !/\s$/.test(current) ? " " : "") + word + " ";
      if (text.length > input.maxLength) return;
      learn(word, true); setDraft(text, text.length);
    }
    function showCorrections(corrections) {
      const draft = input.value, letters = Array.from(draft), rows = [];
      for (const {word,start,end,candidates} of corrections) {
        const key = signature(word,start);
        if (profile().dismissed.has(key) || letters.slice(start,end).join("") !== word) continue;
        const row = make("div", "correction");
        row.append(make("span", "correction-label", candidates.length ? `Correct “${word}”` : `Check spelling of “${word}”`));
        for (const replacement of candidates) row.append(button("correction-choice", replacement, () => {
          const before = letters.slice(0,start).join("");
          const corrected = before + replacement + letters.slice(end).join("");
          if (input.value !== draft || corrected.length > input.maxLength) return;
          const delta = Array.from(replacement).length - (end-start);
          profile().dismissed = new Set([...profile().dismissed].flatMap(prior => {
            const [name,position] = JSON.parse(prior);
            return position === start ? [] : [signature(name, position > start ? position + delta : position)];
          }));
          learn(replacement,true); setDraft(corrected,before.length+replacement.length);
        }));
        row.append(button("correction-keep", "Keep original", () => {
          profile().dismissed.add(key); learn(word, true); row.remove(); focus();
        }));
        rows.push(row);
      }
      el("corrections").replaceChildren(...rows);
    }
    function renderMode() {
      for (const name of ["solo","chat"]) el("mode-"+name).setAttribute("aria-pressed", String(mode===name));
      el("chat-controls").hidden = mode !== "chat";
      el("send-status").textContent = "";
      el("prompts").replaceChildren(...prompts[mode].map(entry => {
        const node = button("prompt", "", () => {
          if (mode === "chat") {
            logs.chat = [{sender:"User A",text:entry[1]}];
            for (const name of ["User A","User B"]) { profiles[name].draft=""; profiles[name].dismissed.clear(); }
            el("speaker").value="User B";
          }
          profile().dismissed.clear();
          setDraft(entry[mode === "solo" ? 1 : 2], entry[mode === "solo" ? 1 : 2].length);
          renderHistory(); renderVocabulary();
        });
        node.append(make("span","prompt-name",entry[0]), make("span","prompt-preview",entry[1]), make("span","prompt-arrow","↗"));
        return node;
      }));
      setDraft(profile().draft); renderHistory(); renderVocabulary();
    }
    function showPrediction(result) {
      preview = result.suggestions.length ? {word:result.suggestions[0].word,mode:result.mode} : null;
      renderPreview(); status(result.warning || "Model ready", result.warning ? "warning" : "ready");
      el("latency").textContent = Number(result.latencyMs).toFixed(1)+" ms";
      el("footprint").textContent = result.footprint;
      el("actual-model").textContent = result.modelLabel || "";
      const choices = result.suggestions.map(({word}) => button("suggestion",word,() => choose(word,result.mode,false)));
      if (result.punctuation) choices.push(button("suggestion suggestion-punct", result.punctuation.word, () => choose(result.punctuation.word,result.mode,true)));
      el("suggestions").replaceChildren(...(choices.length ? choices : [make("span","suggest-empty",input.value ? "No matches. Keep typing." : "Type a message to see suggestions.")]));
      showCorrections(result.corrections || []);
    }
    function showError(message) {
      preview = null; renderPreview(); status("Engine unavailable","error");
      el("latency").textContent = "—";
      el("suggestions").replaceChildren(button("suggestion","Retry",() => schedule(0)));
      el("corrections").replaceChildren(make("span","engine-error",message));
    }
    async function predict(requestVersion) {
      if (busy) { queued=true; return; }
      busy=true;
      try {
        const request = { draft:input.value, context: mode === "chat" ? logs.chat.slice(-2).map(m=>m.text) : [],
          variant:el("variant").value, cacheWeight:Number(el("boost").value), spelling:el("spelling").checked,
          cache:Object.fromEntries(profile().cache), trustedWords:[...profile().trusted] };
        const encoder = new TextEncoder();
        while (encoder.encode(JSON.stringify(request)).length > 128 * 1024) {
          const keys = Object.keys(request.cache);
          if (keys.length) delete request.cache[keys[keys.length - 1]];
          else if (request.trustedWords.length) request.trustedWords.pop();
          else throw new Error("This message is too large to send.");
        }
        const response = await fetch("/api/predict", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(request)});
        if (!response.headers.get("content-type")?.includes("application/json")) throw new Error("Run python web_app.py and open http://127.0.0.1:8765.");
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Prediction failed.");
        if (requestVersion===version) showPrediction(result);
      } catch(error) { if (requestVersion===version) showError(error.message); }
      finally {
        busy=false;
        if (requestVersion===version) el("suggestions").setAttribute("aria-busy","false");
        if (queued) { queued=false; clearTimeout(timer); void predict(version); }
      }
    }
    function schedule(delay=200) {
      preview=null; renderPreview(); clearTimeout(timer); queued=false;
      const requestVersion=++version;
      el("corrections").replaceChildren();
      for (const node of el("suggestions").querySelectorAll("button")) node.disabled=true;
      el("suggestions").setAttribute("aria-busy","true"); status("Predicting…","loading");
      timer=setTimeout(() => void predict(requestVersion),delay);
    }
    const settings=el("settings-dialog");
    el("open-settings").addEventListener("click",()=>settings.showModal());
    el("close-settings").addEventListener("click",()=>settings.close());
    settings.addEventListener("click",event=>{ if(event.target===settings) {
      const r=settings.getBoundingClientRect();
      if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom) settings.close();
    }});
    for(const next of ["solo","chat"]) el("mode-"+next).addEventListener("click",()=>{mode=next;renderMode();});
    el("speaker").addEventListener("change",renderMode);
    el("reset-chat").addEventListener("click",()=>{
      logs.chat=[];
      for(const name of ["User A","User B"]) {profiles[name].draft="";profiles[name].dismissed.clear();}
      el("speaker").value="User A";renderMode();
    });
    el("reset-cache").addEventListener("click",()=>{profile().cache.clear();profile().trusted.clear();renderVocabulary();schedule();});
    input.addEventListener("input",event=>{
      if(event.isComposing) return;
      setDraft(input.value);el("send-status").textContent="";
    });
    input.addEventListener("compositionend",()=>setDraft(input.value));
    input.addEventListener("scroll",renderPreview);document.addEventListener("selectionchange",renderPreview);
    input.addEventListener("keydown",event=>{
      if(event.isComposing||event.keyCode===229) return;
      if(event.key==="Enter") {event.preventDefault();send();}
      if(event.key==="ArrowRight" && !event.shiftKey && preview && !el("draft-preview").hidden) {
        event.preventDefault();choose(preview.word,preview.mode,false);
      }
    });
    el("send").addEventListener("click",send);
    el("boost").addEventListener("input",()=>{el("boost-value").textContent=Number(el("boost").value).toFixed(2);schedule();});
    for(const id of ["variant","spelling"]) el(id).addEventListener("change",()=>schedule());
    renderMode();
  }
  if(document.readyState==="loading") document.addEventListener("DOMContentLoaded",init,{once:true}); else init();
})();
