/**
 * HTR Note-Taking App - Integrated with Air Writing
 * ==================================================
 * 
 * Flow:
 * 1. Create/select note
 * 2. Click "Start Air Writing" button
 * 3. Air writer appends recognized text to current note
 * 4. Auto-saves after each recognition
 */

// ══════════════════════════════════════════════════════════════════════════
// STATE MANAGEMENT
// ══════════════════════════════════════════════════════════════════════════

let currentNoteId = null;
let notes = [];
let airWritingActive = false;

// ══════════════════════════════════════════════════════════════════════════
// DOM ELEMENTS
// ══════════════════════════════════════════════════════════════════════════

const notesList = document.getElementById('notesList');
const noteTextarea = document.getElementById('noteTextarea');
const newNoteBtn = document.getElementById('newNoteBtn');
const startAirWritingBtn = document.getElementById('startAirWritingBtn');
const stopAirWritingBtn = document.getElementById('stopAirWritingBtn');
const deleteNoteBtn = document.getElementById('deleteNoteBtn');
const airWritingPanel = document.getElementById('airWritingPanel');
const editorTitle = document.getElementById('editorTitle');
const wordCount = document.getElementById('wordCount');
const charCount = document.getElementById('charCount');
const lastSaved = document.getElementById('lastSaved');
const recognitionStatus = document.getElementById('recognitionStatus');

// ══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    console.log('HTR Notes App initialized');
    
    // Load existing notes
    loadNotes();
    
    // Setup event listeners
    newNoteBtn.addEventListener('click', createNewNote);
    startAirWritingBtn.addEventListener('click', startAirWriting);
    stopAirWritingBtn.addEventListener('click', stopAirWriting);
    deleteNoteBtn.addEventListener('click', deleteCurrentNote);
    noteTextarea.addEventListener('input', handleTextChange);
    
    // Poll for air writing results
    startPollingForResults();
});

// ══════════════════════════════════════════════════════════════════════════
// NOTES MANAGEMENT
// ══════════════════════════════════════════════════════════════════════════

async function loadNotes() {
    try {
        const response = await fetch('/api/notes');
        const data = await response.json();
        
        if (data.success) {
            notes = data.notes;
            renderNotes();
        }
    } catch (error) {
        console.error('Failed to load notes:', error);
    }
}

function renderNotes() {
    if (notes.length === 0) {
        notesList.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📝</div>
                <p>No notes yet</p>
                <p style="font-size: 12px; margin-top: 4px;">Click "New Note" to start</p>
            </div>
        `;
        return;
    }

    notesList.innerHTML = notes.map(note => `
        <div class="note-item ${note.id === currentNoteId ? 'active' : ''}" onclick="selectNote(${note.id})">
            <div class="note-preview">${note.text.substring(0, 50) || 'Empty note'}</div>
            <div class="note-meta">
                <span>${new Date(note.created_at).toLocaleDateString()}</span>
                <span>${note.confidence ? (note.confidence * 100).toFixed(0) + '%' : ''}</span>
            </div>
        </div>
    `).join('');
}

async function createNewNote() {
    try {
        const response = await fetch('/api/notes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: '', confidence: 0 })
        });

        const data = await response.json();
        if (data.success) {
            notes.unshift(data.note);
            selectNote(data.note.id);
            renderNotes();
        }
    } catch (error) {
        console.error('Failed to create note:', error);
    }
}

function selectNote(noteId) {
    currentNoteId = noteId;
    const note = notes.find(n => n.id === noteId);
    
    if (note) {
        noteTextarea.value = note.text;
        noteTextarea.disabled = false;
        startAirWritingBtn.disabled = false;
        deleteNoteBtn.disabled = false;
        editorTitle.textContent = `Note #${note.id}`;
        updateStats();
        renderNotes();
    }
}

async function deleteCurrentNote() {
    if (!currentNoteId) return;
    if (!confirm('Delete this note?')) return;

    try {
        const response = await fetch(`/api/notes/${currentNoteId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            notes = notes.filter(n => n.id !== currentNoteId);
            currentNoteId = null;
            noteTextarea.value = '';
            noteTextarea.disabled = true;
            startAirWritingBtn.disabled = true;
            deleteNoteBtn.disabled = true;
            editorTitle.textContent = 'Select a note or create a new one';
            renderNotes();
            updateStats();
        }
    } catch (error) {
        console.error('Failed to delete note:', error);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// TEXT EDITING
// ══════════════════════════════════════════════════════════════════════════

function handleTextChange() {
    updateStats();
    saveCurrentNote();
}

function updateStats() {
    const text = noteTextarea.value;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = text.length;

    wordCount.textContent = `${words} word${words !== 1 ? 's' : ''}`;
    charCount.textContent = `${chars} character${chars !== 1 ? 's' : ''}`;
}

let saveTimeout;
async function saveCurrentNote() {
    if (!currentNoteId) return;

    clearTimeout(saveTimeout);
    saveTimeout = setTimeout(async () => {
        try {
            const response = await fetch(`/api/notes/${currentNoteId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: noteTextarea.value })
            });

            if (response.ok) {
                const note = notes.find(n => n.id === currentNoteId);
                if (note) note.text = noteTextarea.value;
                renderNotes();
                lastSaved.textContent = 'Saved ' + new Date().toLocaleTimeString();
            }
        } catch (error) {
            console.error('Failed to save note:', error);
        }
    }, 500);
}

// ══════════════════════════════════════════════════════════════════════════
// AIR WRITING INTEGRATION
// ══════════════════════════════════════════════════════════════════════════

async function startAirWriting() {
    if (!currentNoteId) {
        alert('Please select a note first!');
        return;
    }
    
    airWritingActive = true;
    airWritingPanel.classList.add('active');
    recognitionStatus.textContent = 'Launching camera...';
    
    // Signal backend to start air writing and launch camera
    try {
        const response = await fetch('/api/air-writing/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ note_id: currentNoteId })
        });
        
        const result = await response.json();
        
        if (result.success) {
            if (result.camera_launched) {
                recognitionStatus.textContent = 'Camera launched! Waiting for input...';
                console.log('✓ Camera opened automatically (PID:', result.pid, ')');
            } else {
                recognitionStatus.textContent = 'Ready (run air_writer_webapp.py manually)';
                console.log('⚠ Camera not launched. Run manually: python air_writer_webapp.py');
            }
        } else {
            recognitionStatus.textContent = 'Failed to start';
            console.error('Failed to start air writing:', result.error);
        }
    } catch (error) {
        recognitionStatus.textContent = 'Error';
        console.error('Failed to start air writing:', error);
    }
}

async function stopAirWriting() {
    airWritingActive = false;
    airWritingPanel.classList.remove('active');
    recognitionStatus.textContent = 'Stopped';
    
    // Signal backend to stop
    try {
        await fetch('/api/air-writing/stop', {
            method: 'POST'
        });
        
        console.log('Air writing stopped');
    } catch (error) {
        console.error('Failed to stop air writing:', error);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// POLLING FOR AIR WRITING RESULTS
// ══════════════════════════════════════════════════════════════════════════

let lastRecognitionId = 0;

async function startPollingForResults() {
    // Poll every 500ms for new recognitions
    setInterval(async () => {
        if (!airWritingActive || !currentNoteId) return;
        
        try {
            const response = await fetch(`/api/air-writing/poll?last_id=${lastRecognitionId}`);
            const data = await response.json();
            
            if (data.success && data.recognitions && data.recognitions.length > 0) {
                // Process each new recognition
                data.recognitions.forEach(recognition => {
                    appendRecognizedText(recognition.text, recognition.confidence);
                    lastRecognitionId = recognition.id;
                });
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 500);
}

function appendRecognizedText(text, confidence) {
    if (!currentNoteId || !text) return;
    
    // Append text to textarea (add space if not empty)
    const currentText = noteTextarea.value;
    const newText = currentText + (currentText ? ' ' : '') + text;
    noteTextarea.value = newText;
    
    // Update UI
    updateStats();
    saveCurrentNote();
    
    // Update status
    recognitionStatus.textContent = `"${text}" (${(confidence * 100).toFixed(0)}%)`;
    
    // Flash feedback
    noteTextarea.style.transition = 'background 0.3s ease';
    noteTextarea.style.background = 'rgba(59, 130, 246, 0.1)';
    setTimeout(() => {
        noteTextarea.style.background = 'transparent';
    }, 300);
    
    console.log(`Appended: "${text}" with ${(confidence * 100).toFixed(1)}% confidence`);
}

// ══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════

// Make selectNote available globally for onclick
window.selectNote = selectNote;

console.log('app.js loaded successfully');