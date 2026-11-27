-- spp_importer.lua - bring a converted Substance 3D Painter project into PawPaint.
--
-- 1. Convert the .spp with SppToPawPaint.exe (drag the .spp onto it). That
--    writes a folder named "<project> (PawPaint import)" next to the .spp and
--    tells this plugin about it.
-- 2. In PawPaint: Plugins > Substance Importer. The latest conversion is
--    already filled in; click Import.
--
-- Mesh sources: the mesh rebuilt from the .spp, your own model file (keeps
-- shape keys), or the project that is open right now. Texture sets are matched
-- to materials by name; anything left over can be matched by hand. Saving is
-- not scriptable, so use File > Save As when it finishes.
--
-- Unofficial community tool, not made or supported by PawWorks or Adobe.

local IMPORTER_VERSION = "0.7.0"
local PENDING_FILE = "scripts/spp_importer_pending.json"

local MESH_SPP, MESH_OWN, MESH_OPEN = 1, 2, 3
-- 'Size' option: the size of the textures in the PawPaint project. PawPaint
-- keeps every layer on the graphics card, and what a layer takes goes with
-- the texture area, so a material that doesn't fit at 4096 needs a quarter of
-- the video memory at 2048. Auto starts at the sizes of the conversion and
-- makes a material a size smaller when its layers don't fit, instead of
-- leaving layers out. PawPaint scales the converted images itself.
--
-- What this PawPaint can do. 1.28 added the calls below; on an older PawPaint
-- the plugin works as it did before (one size for the whole project, a
-- refused import instead of a question, layers found again by name, a
-- solid-colour image where a value can't be set, no Browse buttons).
local function has(t, name)
    return type(t) == "table" and type(t[name]) == "function"
end
local CAN = {
    materialSize = has(pp.project, "setMaterialResolution") and has(pp.project, "materialResolution"),
    estimate = has(pp.import, "estimate"),     -- 'would this import fit?' without importing
    ids = has(pp.layers, "indexOf"),           -- the import says which layers it made
    values = has(pp.fill, "setValue"),         -- a value for emission and custom channels
    dialog = type(pp.dialog) == "table" and has(pp.dialog, "selectFolder") and has(pp.dialog, "openFile"),
}
local SIZE_AUTO, SIZE_CONVERTED = 1, 2
local SIZE_OPTIONS = { "Auto (fit graphics card)", "As converted", "2048", "1024", "512" }
local SIZE_VALUES = { [3] = 2048, [4] = 1024, [5] = 512 }
local MIN_AUTO_SIZE = 1024   -- Auto doesn't go below this by itself
-- kinds for channels a conversion may use beyond the built-in ones
local EXTRA_CHANNEL_KINDS = {opacity = "scalar", specular = "color", glossiness = "scalar", displacement = "signed"}

local state = {
    folder = "",
    mesh = "",
    meshSource = MESH_SPP,
    size = SIZE_AUTO,
    flatMode = 1,          -- 1 on top + visible, 2 on top + hidden, 3 leave out
    layers = true,
    baked = true,
    height = true,         -- false: nothing that is imported changes the height (bumps)
    force = false,
    status = "Convert a .spp with SppToPawPaint, then click Import.",
    busy = false,
    pendingText = nil,     -- "Latest conversion: ..." line
    mapping = nil,         -- { sets = {...}, choices = {...}, options = {...}, continue = fn }
    jobId = nil,
    cancelled = false,
}

local panelName = "Substance Importer"
local buildPanel -- forward

local function log(msg)
    pp.log("[spp] " .. msg)
end

local function setStatus(msg)
    state.status = msg
    log(msg)
    buildPanel()
end

-- -- paths ------------------------------------------------------------------
local function trim(s)
    s = s or ""
    s = s:gsub("^%s+", ""):gsub("%s+$", "")
    s = s:gsub('^"(.*)"$', "%1"):gsub("^'(.*)'$", "%1")
    return s
end

local function isAbsolute(p)
    return p:sub(1, 1) == "/" or p:sub(1, 1) == "\\" or p:match("^%a:[/\\]") ~= nil
end

local function join(dir, p)
    if isAbsolute(p) then return p end
    local last = dir:sub(-1)
    if last == "/" or last == "\\" then return dir .. p end
    return dir .. "/" .. p
end

-- Accepts the import folder, its import.json, or the original .spp.
local function resolveFolder(p)
    p = trim(p)
    if p == "" then return nil end
    local lower = p:lower()
    if lower:sub(-12) == "\\import.json" or lower:sub(-12) == "/import.json" then
        return p:sub(1, -13)
    end
    if lower:sub(-4) == ".spp" then
        return p:sub(1, -5) .. " (PawPaint import)"
    end
    return (p:gsub("[/\\]+$", ""))
end

local function nonNull(v)
    if v == pp.json.null then return nil end
    return v
end

-- Loose name match: case, spaces, underscores, dashes and ".001" suffixes ignored.
local function norm(s)
    s = tostring(s or ""):lower()
    s = s:gsub("%.%d+$", "")
    s = s:gsub("[^%w]", "")
    return s
end

-- -- latest conversion handoff -------------------------------------------------
local function readPending()
    local p = pp.json.readFile(PENDING_FILE)
    if type(p) ~= "table" or p.format ~= "spp2pawpaint-pending" then return nil end
    return p
end

local function usePending(quiet)
    local p = readPending()
    if not p or type(p.folder) ~= "string" then
        if not quiet then setStatus("No conversion found yet. Run SppToPawPaint on a .spp first.") end
        return false
    end
    state.folder = p.folder
    local when = nonNull(p.createdText) or "earlier"
    state.pendingText = string.format("Latest conversion: %s (%s)", tostring(nonNull(p.source) or "?"), when)
    if not quiet then setStatus("Ready - click Import.") else state.status = "Ready - click Import." end
    return true
end

-- -- building the import spec -----------------------------------------------
-- level: 0 everything chosen in the panel; 1 leaves out the baked normal
-- layer; 2 also the hidden layers (they don't show, but they are the options
-- a creator left in the file); 3 only the flattened Substance result. Levels
-- 1 to 3 are the fallbacks when PawPaint refuses a texture set for lack of
-- video memory and the project can't be made smaller instead.
local LAST_LEVEL = 3
local HEIGHT_CHANNELS = { height = true, displacement = true }

local function wantLayer(l, level, hidden)
    level = level or 0
    local spp = nonNull(l.spp) or {}
    local role = nonNull(spp.role)
    if role == "flattened" then return state.flatMode ~= 3 end
    if level >= LAST_LEVEL then return false end
    if role == "baked_normal" then return state.baked and level < 1 end
    if level >= 2 and hidden then return false end
    return state.layers
end

-- Returns spec, meta (layer info by its place in spec, and by its tag) and
-- the number of hidden layers left out, for one bundle.
local function buildSpec(dir, manifest, level)
    local spec, meta = {}, {}
    local newIndex = {}
    local hiddenAt, leftHidden = {}, 0
    for i, l in ipairs(nonNull(manifest.layers) or {}) do
        -- hidden itself, or inside a hidden folder (folders come before their layers)
        local up = nonNull(l.parent)
        hiddenAt[i] = nonNull(l.visible) == false or (up ~= nil and hiddenAt[up] == true)
        local want = wantLayer(l, level, hiddenAt[i])
        if not want and hiddenAt[i] and wantLayer(l, 1, false) then leftHidden = leftHidden + 1 end
        if want then
            local spp = nonNull(l.spp) or {}
            local tag = nonNull(spp.tag) or ("#spp" .. i)
            local name = tostring(nonNull(l.name) or "Layer")
            if #name > 200 then name = name:sub(1, 200) end
            -- A value PawPaint can set itself needs no solid-colour image.
            local solid = CAN.values and nonNull(l.type) == "Fill" and nonNull(spp.solid) or nil
            if type(solid) ~= "table" then solid = nil end
            local s = {
                type = nonNull(l.type),
                -- without layer ids the layers are found again by this tag in their name
                name = CAN.ids and name or (name .. " " .. tag),
                visible = nonNull(l.visible),
                opacity = nonNull(l.opacity),
                blendMode = nonNull(l.blendMode),
                normalConvention = nonNull(l.normalConvention),
            }
            if nonNull(spp.role) == "flattened" and state.flatMode == 2 then
                s.visible = false
            end
            local parent = nonNull(l.parent)
            if parent then s.parent = newIndex[parent] end -- nil if the folder was left out
            local ov = nonNull(l.channelOverrides)
            if type(ov) == "table" then
                s.channelOverrides = {}
                for id, e in pairs(ov) do
                    e = nonNull(e)
                    if type(e) == "table" and (state.height or not HEIGHT_CHANNELS[id]) then
                        s.channelOverrides[id] = { blendMode = nonNull(e.blendMode), opacity = nonNull(e.opacity) }
                    end
                end
                if next(s.channelOverrides) == nil then s.channelOverrides = nil end
            end
            local ch = nonNull(l.channels)
            if type(ch) == "table" then
                s.channels = {}
                for id, p in pairs(ch) do
                    p = nonNull(p)
                    -- 'Height' unticked: no layer brings a height image
                    if type(p) == "string" and (state.height or not HEIGHT_CHANNELS[id])
                            and not (solid and nonNull(solid[id]) ~= nil) then
                        s.channels[id] = join(dir, p)
                    end
                end
                if next(s.channels) == nil then s.channels = nil end
            end
            local masks = nonNull(l.masks)
            if type(masks) == "table" then
                s.masks = {}
                for _, mk in ipairs(masks) do
                    mk = nonNull(mk)
                    if type(mk) == "table" and type(nonNull(mk.path)) == "string" then
                        s.masks[#s.masks + 1] = {
                            path = join(dir, mk.path), name = nonNull(mk.name),
                            blendMode = nonNull(mk.blendMode), enabled = nonNull(mk.enabled),
                            invert = nonNull(mk.invert),
                        }
                    end
                end
                if #s.masks == 0 then s.masks = nil end
            end
            spec[#spec + 1] = s
            newIndex[i] = #spec
            local info = { name = name, type = s.type, fill = nonNull(spp.fill),
                           disabled = nonNull(spp.disabledChannels), solid = solid,
                           only = nonNull(spp.onlyChannels), maps = nonNull(spp.mapChannels) }
            meta[tag] = info
            meta[#spec] = info
        end
    end
    return spec, meta, leftHidden
end

local function gigabytes(mb)
    return string.format("%.1f GB", mb / 1024)
end

-- "it needs about 15.1 GB and 7.9 GB is free", from PawPaint's refusal.
local function memoryText(need, free)
    if need and free then
        return string.format("it needs about %s of video memory and %s is free", gigabytes(need), gigabytes(free))
    end
    return "there is not enough free video memory"
end

-- -- after a material's layers land -------------------------------------------
local FILL_CHANNELS = { "albedo", "roughness", "metallic", "normal", "height" }

local function asSet(list)
    local s = {}
    for _, v in ipairs(list or {}) do s[v] = true end
    return s
end

local function applyFill(i, fill, mapChannels)
    fill = fill or {}
    local maps = asSet(mapChannels)
    local function v(id)
        local e = nonNull(fill[id])
        if type(e) ~= "table" then return nil end
        return nonNull(e.value)
    end
    local c = v("albedo")
    if type(c) == "table" then pp.fill.setAlbedo(i, c[1], c[2], c[3]) end
    local r = v("roughness")
    if type(r) == "number" then pp.fill.setRoughness(i, r) end
    local m = v("metallic")
    if type(m) == "number" then pp.fill.setMetallic(i, m) end
    local n = v("normal")
    if type(n) == "table" then pp.fill.setNormal(i, n[1], n[2], n[3]) end
    local h = v("height")
    if not state.height then h = nil end
    if type(h) == "number" then pp.fill.setHeight(i, h) end
    -- A fill layer multiplies its channel images by the fill value (colour
    -- images by the colour, grey ones through a level where 0.5 is neutral,
    -- height by a strength where 1 is neutral). New fill layers start grey,
    -- which would darken imported images, so make the value neutral.
    if maps.albedo and type(c) ~= "table" then pp.fill.setAlbedo(i, 1, 1, 1) end
    if maps.roughness and type(r) ~= "number" then pp.fill.setRoughness(i, 0.5) end
    if maps.metallic and type(m) ~= "number" then pp.fill.setMetallic(i, 0.5) end
    if maps.height and type(h) ~= "number" then pp.fill.setHeight(i, 1.0) end
    for _, id in ipairs(FILL_CHANNELS) do
        if not maps[id] then
            pp.fill.setChannelEnabled(i, id, fill[id] ~= nil and nonNull(fill[id]) ~= nil)
        end
    end
    -- 'Height' unticked: fill layers leave the height alone too
    if not state.height then pp.fill.setChannelEnabled(i, "height", false) end
end

-- The values behind a fill's solid-colour images (emission, custom channels),
-- set directly: {channel = number or {r, g, b}}.
local function applySolid(i, solid)
    for id, v in pairs(solid) do
        v = nonNull(v)
        local ok = false
        if type(v) == "number" then
            ok = pp.fill.setValue(i, id, v)
        elseif type(v) == "table" then
            ok = pp.fill.setValue(i, id, v[1], v[2], v[3])
        end
        if ok ~= false then pp.fill.setChannelEnabled(i, id, true) end
    end
end

local function finishLayer(i, info)
    local isFill = pp.layers.type(i) == "Fill"
    if isFill and (info.fill or info.maps or not state.height) then
        applyFill(i, info.fill, info.maps)
    end
    if isFill and info.solid then applySolid(i, info.solid) end
    if isFill and type(info.only) == "table" then
        local only = asSet(info.only)
        for _, id in ipairs(FILL_CHANNELS) do
            if not only[id] then pp.fill.setChannelEnabled(i, id, false) end
        end
    end
    if type(info.disabled) == "table" then
        for _, id in ipairs(info.disabled) do
            if isFill then pp.fill.setChannelEnabled(i, id, false)
            else pp.layers.setChannelOpacity(i, id, 0) end
        end
    end
end

-- ids: the ids of the layers the import made, in the order of its spec
-- (PawPaint 1.28+), or nil; then the layers are found by the tag in their name.
local function finishMaterial(matIndex, before, meta, removeDefaults, ids)
    pp.layers.setActiveMaterial(matIndex)
    local count = pp.layers.count()
    local found, leftovers = 0, {}
    if ids then
        local mine = {}
        for k = 1, #ids do
            local id = ids[k]
            local i = (type(id) == "number" and id ~= 0) and pp.layers.indexOf(id) or -1
            if i >= 0 then
                mine[i] = true
                if meta[k] then
                    finishLayer(i, meta[k])
                    found = found + 1
                end
            end
        end
        for i = count - 1, 0, -1 do
            if not mine[i] then leftovers[#leftovers + 1] = i end
        end
    else
        for i = count - 1, 0, -1 do
            local nm = pp.layers.name(i) or ""
            local tag = nm:match(" (#spp%w+)$")
            local info = tag and meta[tag]
            if info then
                finishLayer(i, info)
                pp.layers.setName(i, info.name)
                found = found + 1
            elseif not tag then
                leftovers[#leftovers + 1] = i
            end
        end
    end
    -- In a fresh project, remove the default layer(s) so the stack is exactly
    -- what came from Substance. Never in a project the user already had open.
    if removeDefaults and #leftovers > 0 and #leftovers <= before then
        for _, i in ipairs(leftovers) do pp.layers.remove(i) end -- highest index first
    end
    return found
end

-- -- the import ------------------------------------------------------------------
local function importMaterials(ctx, done)
    if state.cancelled then
        done("Cancelled. Layers already imported stay; use Ctrl+Z to undo them.")
        return
    end
    if #ctx.queue == 0 then
        done(nil)
        return
    end
    local job = table.remove(ctx.queue, 1)
    ctx.step = ctx.step + 1
    local dir = join(ctx.folder, job.entry.bundle)
    local manifest, why = pp.json.readFile(join(dir, "manifest.json"))
    if not manifest then
        log("could not read " .. job.entry.bundle .. "/manifest.json: " .. tostring(why))
        importMaterials(ctx, done)
        return
    end
    local have = {}
    for _, c in ipairs(pp.channels.list()) do have[c.id] = true end
    for _, c in ipairs(nonNull(manifest.channels) or {}) do
        if not have[c.id] then
            if pp.channels.add(c.id, nonNull(c.name) or c.id, nonNull(c.kind) or "scalar") == false then
                log("could not add channel " .. tostring(c.id))
            end
            have[c.id] = true
        end
    end
    -- A layer naming a channel PawPaint doesn't have makes it refuse the whole
    -- texture set, so add any undeclared channel, or drop it from that layer.
    for _, L in ipairs(nonNull(manifest.layers) or {}) do
        for _, key in ipairs({"channels", "channelOverrides"}) do
            local t = nonNull(L[key])
            if type(t) == "table" then
                for cid in pairs(t) do
                    if not have[cid] then
                        local kind = EXTRA_CHANNEL_KINDS[cid] or "scalar"
                        local name = cid:sub(1, 1):upper() .. cid:sub(2)
                        if pp.channels.add(cid, name, kind) == false then
                            log("could not add channel '" .. tostring(cid) .. "'; it was left out")
                            have[cid] = false
                        else
                            have[cid] = true
                        end
                    end
                    if have[cid] == false then t[cid] = nil end
                end
            end
        end
    end
    pp.layers.setActiveMaterial(job.index)
    -- This material's own size, in a project this import made (PawPaint 1.28+).
    if ctx.setSize and job.size and not ctx.setSize(job, job.size) then
        job.size = nil      -- it stays at the project's size
    end
    local before = pp.layers.count()
    local full = 0
    local short = nil      -- memoryText of the first refusal
    local advice = "Close other programs that use the graphics card (Substance Painter, games, browsers)"
        .. (ctx.canResize and ", or pick a smaller Size, then import again." or ", then import again.")
    local attempt
    -- The layers don't fit the free video memory (need, free: MB, or nil when
    -- PawPaint didn't say): make this material a size smaller if that is
    -- allowed, else the whole project, else leave something out.
    local function tooBig(level, need, free)
        short = short or memoryText(need, free)
        if level == 0 and ctx.auto and ctx.setSize and job.size and job.size > MIN_AUTO_SIZE then
            local from = job.size
            local to = math.floor(from / 2)
            if ctx.setSize(job, to) then
                job.size = to
                job.shrunk = job.shrunk or { from = from, layers = full, why = memoryText(need, free) }
                log(string.format("'%s' doesn't fit at %d (%s); importing this material at %d instead, with every layer",
                    job.entry.name, from, memoryText(need, free), to))
                attempt(0)
                return
            end
        end
        -- Before 1.28: the whole project one size smaller, which also keeps
        -- every layer (a quarter of the memory per step).
        if level == 0 and ctx.shrink and ctx.size > MIN_AUTO_SIZE then
            ctx.shrink(job.entry.name, full, need, free)
            return
        end
        -- Otherwise try again with less instead of leaving it empty.
        if level < LAST_LEVEL then
            local nextLevel = level + 1
            log(string.format("'%s': trying again with %s", job.entry.name,
                nextLevel == 1 and "the baked normal layer left out"
                or nextLevel == 2 and "the hidden layers left out too"
                or "only the flattened Substance result"))
            attempt(nextLevel)
            return
        end
        importMaterials(ctx, done)
    end
    attempt = function(level)
        local spec, meta, leftHidden = buildSpec(dir, manifest, level)
        if level == 0 then full = #spec end
        if #spec == 0 then
            if level > 0 then
                ctx.notes[#ctx.notes + 1] = string.format("'%s' was left out: not enough free video memory", job.entry.name)
            end
            importMaterials(ctx, done)
            return
        end
        -- Ask first (PawPaint 1.28+): the import's own check, nothing is queued.
        if CAN.estimate then
            local est = pp.import.estimate(job.index, spec)
            if type(est) == "table" and est.fits == false then
                local need = tonumber(est.bytesNeeded) and est.bytesNeeded / 1048576 or nil
                local free = tonumber(est.bytesFree) and est.bytesFree / 1048576 or nil
                log(string.format("'%s' would not fit: %s", job.entry.name, memoryText(need, free)))
                tooBig(level, need, free)
                return
            end
        end
        setStatus(string.format("Importing '%s' into '%s' (%d of %d, %d layers%s)...", job.entry.name,
            tostring(pp.mesh.materialName(job.index)), ctx.step, ctx.total, #spec,
            job.size and string.format(", %d x %d", job.size, job.size) or ""))
        local id, err = pp.import.layers(job.index, spec, function(r)
            state.jobId = nil
            if r.ok then
                local ids = nil
                if CAN.ids then
                    ids = nonNull(r.layerIds)
                    if type(ids) ~= "table" then
                        ids = {}
                        log(string.format("'%s': PawPaint did not say which layers it made, so their fill values "
                            .. "could not be set", job.entry.name))
                    end
                end
                local n = finishMaterial(job.index, before, meta, ctx.fresh, ids)
                log(string.format("'%s': %d layers imported", job.entry.name, n))
                if job.shrunk then
                    ctx.notes[#ctx.notes + 1] = string.format(
                        "'%s' was imported at %d instead of %d: at %d its %d layers didn't fit this graphics card (%s). "
                        .. "Every layer is there. To keep %d, set Size to 'As converted'; a material that doesn't fit "
                        .. "then comes in without some of its layers.",
                        job.entry.name, job.size, job.shrunk.from, job.shrunk.from, job.shrunk.layers, job.shrunk.why,
                        job.shrunk.from)
                end
                -- say what a fallback left out
                if level == 1 and #spec < full then
                    ctx.notes[#ctx.notes + 1] = string.format(
                        "'%s': not enough free video memory for everything (%s), so the baked normal layer was left out. %s",
                        job.entry.name, short, advice)
                elseif level == 2 then
                    ctx.notes[#ctx.notes + 1] = string.format(
                        "'%s': not enough free video memory for its %d layers (%s), so its %d hidden layers%s were left out. %s",
                        job.entry.name, full, short, leftHidden, state.baked and " and the baked normal" or "", advice)
                elseif level >= LAST_LEVEL then
                    ctx.notes[#ctx.notes + 1] = string.format(
                        "'%s': not enough free video memory for its %d layers (%s), so only the flattened result was imported. %s",
                        job.entry.name, full, short, advice)
                end
            else
                log(string.format("'%s': import %s", job.entry.name, tostring(r.state)))
            end
            for _, f in ipairs(r.leftOut or {}) do
                log("  left out " .. tostring(f.path) .. " (" .. tostring(f.why) .. ")")
            end
            importMaterials(ctx, done)
        end)
        if not id then
            local why = tostring(err)
            log(string.format("'%s' was refused: %s", job.entry.name, why))
            -- PawPaint 1.27+ refuses an import that needs more video memory
            -- than is free.
            if why:lower():find("video memory", 1, true) then
                tooBig(level, tonumber(why:match("needs about (%d+) MB")), tonumber(why:match("about (%d+) MB is free")))
                return
            end
            importMaterials(ctx, done)
        else
            state.jobId = id
        end
    end
    attempt(0)
end

-- sizing (nil when importing into the open project): { size = the project's
-- size now, converted = the conversion's size, auto = true when the size may
-- be lowered, reopen = function(size, onOpen) }.
local function runImport(info, folder, fresh, sizing)
    local entries = nonNull(info.materials) or {}
    local byNorm = {}
    for _, e in ipairs(entries) do byNorm[norm(e.name)] = e end
    local queue, used, unmatched = {}, {}, {}
    local n = pp.mesh.materialCount()
    for i = 0, n - 1 do
        local e = byNorm[norm(pp.mesh.materialName(i))]
        if e and not used[e] then
            queue[#queue + 1] = { index = i, entry = e }
            used[e] = true
        end
    end
    local usedIndex = {}
    for _, q in ipairs(queue) do usedIndex[q.index] = true end
    local freeMaterials = 0
    for i = 0, n - 1 do
        if not usedIndex[i] then freeMaterials = freeMaterials + 1 end
    end
    local skipped = {}
    for _, e in ipairs(entries) do
        if not used[e] then
            -- Texture sets with no part in the .spp's own model have nowhere to
            -- go on that model; and if every material is already taken there is
            -- nothing to choose from. Skip those instead of asking.
            if (state.meshSource == MESH_SPP and nonNull(e.inMesh) == false) or freeMaterials == 0 then
                skipped[#skipped + 1] = tostring(e.name)
            else
                unmatched[#unmatched + 1] = e
            end
        end
    end
    if #skipped > 0 then
        log("skipped (not used by any part of the model): " .. table.concat(skipped, ", "))
    end

    local shrunk = nil     -- why the project is smaller than the conversion, once it is
    local function go()
        pp.setTimeLimit(600)
        local jobs = {}
        for i, q in ipairs(queue) do
            jobs[i] = q
            -- the size this material gets: the texture set's own, at most the project's
            q.shrunk = nil
            q.size = sizing and math.min(tonumber(nonNull(q.entry.resolution)) or sizing.converted, sizing.size) or nil
        end
        local ctx = { folder = folder, queue = jobs, total = #jobs, step = 0, fresh = fresh, notes = {},
                      size = sizing and sizing.size or 0, canResize = sizing ~= nil,
                      auto = sizing ~= nil and sizing.auto }
        if sizing and fresh and CAN.materialSize then
            -- every material at its own size; never in a project the user already had open
            ctx.setSize = function(job, size)
                if pp.project.materialResolution(job.index) == size then return true end
                return pp.project.setMaterialResolution(job.index, size) == true
            end
        elseif sizing and sizing.auto and fresh then
            ctx.shrink = function(name, layers, need, free)
                local from = sizing.size
                sizing.size = math.floor(from / 2)
                shrunk = shrunk or string.format(
                    "at %d '%s' with its %d layers didn't fit this graphics card (%s)",
                    from, tostring(name), layers, memoryText(need, free))
                log(string.format("'%s' doesn't fit at %d; importing the whole project at %d instead, with every layer",
                    tostring(name), from, sizing.size))
                state.jobId = nil
                sizing.reopen(sizing.size, go)
            end
        end
        if shrunk then
            ctx.notes[#ctx.notes + 1] = string.format(
                "Imported at %d instead of %d: %s. Every layer is there. To keep %d, set Size to 'As converted'; "
                .. "texture sets that don't fit then come in without some of their layers.",
                sizing.size, sizing.converted, shrunk, sizing.converted)
        end
        local sizeNote = nonNull(info.sizeNote)
        if type(sizeNote) == "string" then ctx.notes[#ctx.notes + 1] = sizeNote end
        if nonNull(info.partial) == true then
            -- the converter was closed before the end: it saves each texture set as it finishes
            ctx.notes[#ctx.notes + 1] = string.format(
                "This conversion was stopped early: %d of %s texture sets are in this folder. "
                .. "Convert the .spp again for the rest.",
                #(nonNull(info.materials) or {}), tostring(nonNull(info.setsTotal) or "all"))
        end
        if ctx.total == 0 then
            state.busy = false
            setStatus("Nothing to import: no texture set is matched to a material.")
            return
        end
        importMaterials(ctx, function(stopped)
            state.busy = false
            state.cancelled = false
            pp.layers.setActiveMaterial(queue[1] and queue[1].index or 0)
            if stopped then setStatus(stopped) return end
            local msg = "Done. Use File > Save As to keep it."
            if sizing then
                local sizes, differ = {}, false
                for _, q in ipairs(queue) do
                    if q.size then
                        sizes[#sizes + 1] = string.format("%s %d", tostring(q.entry.name), q.size)
                        if q.size ~= sizing.size then differ = true end
                    end
                end
                if ctx.setSize and differ then
                    msg = string.format("Done, each material at its own size (%s). Use File > Save As to keep it.",
                        table.concat(sizes, ", "))
                else
                    msg = string.format("Done, at %d x %d. Use File > Save As to keep it.", sizing.size, sizing.size)
                end
            end
            for _, line in ipairs(ctx.notes) do
                log(line)
                msg = msg .. " " .. line
            end
            local w = nonNull(info.warnings) or {}
            if #w > 0 then
                msg = msg .. string.format(" %d conversion notes are in report.txt and the log.", #w)
                for _, line in ipairs(w) do log("note: " .. tostring(line)) end
            end
            setStatus(msg)
        end)
    end

    if #unmatched == 0 then
        go()
        return
    end
    -- let the user match the rest by hand
    local options, idxOf = { "(skip)" }, {}
    for i = 0, n - 1 do
        options[#options + 1] = tostring(pp.mesh.materialName(i))
        idxOf[#options] = i
    end
    local choices = {}
    for k, e in ipairs(unmatched) do choices[k] = 1 end
    state.mapping = {
        sets = unmatched, options = options, choices = choices,
        continue = function()
            for k, e in ipairs(unmatched) do
                local i = idxOf[choices[k]]
                if i then queue[#queue + 1] = { index = i, entry = e } end
            end
            state.mapping = nil
            go()
        end,
    }
    setStatus(string.format("%d texture set(s) didn't match a material by name. Pick one for each below, then Continue.",
        #unmatched))
end

local function startImport()
    if state.busy then
        setStatus("An import is already running.")
        return
    end
    if trim(state.folder) == "" then usePending(true) end
    local folder = resolveFolder(state.folder)
    if not folder then
        setStatus("No conversion selected. Run SppToPawPaint on a .spp, or paste its folder.")
        return
    end
    local info, why = pp.json.readFile(join(folder, "import.json"))
    if not info then
        setStatus("No import.json in that folder. Convert the .spp with SppToPawPaint first. (" .. tostring(why) .. ")")
        return
    end
    if info.format ~= "spp2pawpaint-import" then
        setStatus("That import.json was not written by SppToPawPaint.")
        return
    end
    if type(info.version) ~= "number" or info.version > 1 then
        setStatus("This folder needs a newer Substance Importer plugin.")
        return
    end
    state.busy = true
    state.cancelled = false

    if state.meshSource == MESH_OPEN then
        if not pp.mesh.isLoaded() then
            state.busy = false
            setStatus("No model is open. Open your model first, or pick another mesh source.")
            return
        end
        runImport(info, folder, false)
        return
    end

    local mesh
    if state.meshSource == MESH_OWN then
        mesh = trim(state.mesh)
        if mesh == "" then
            state.busy = false
            setStatus("Type the path of your model file in 'Model', or pick another mesh source.")
            return
        end
    else
        mesh = join(folder, nonNull(info.mesh) or "model.glb")
    end
    local converted = tonumber(info.resolution) or 2048
    local sizing = { converted = converted, size = converted, auto = state.size == SIZE_AUTO }
    local fixed = SIZE_VALUES[state.size]
    if fixed then sizing.size = math.min(fixed, converted) end   -- bigger than the conversion adds nothing
    local function open(size, force, onOpen)
        setStatus(string.format("Opening %s at %d x %d ...", mesh, size, size))
        local ok, err = pp.project.openMesh(mesh, { resolution = size, force = force },
            function(r)
                if not r.ok then
                    state.busy = false
                    setStatus("Could not open the model: " .. tostring(r.error))
                    return
                end
                onOpen()
            end)
        if not ok then
            state.busy = false
            local e = tostring(err)
            if e:lower():find("unsaved") then
                e = e .. " - save your project first, or tick 'Discard unsaved changes'."
            end
            setStatus("Could not start: " .. e)
        end
    end
    -- Opening again at a smaller size replaces the project this import has
    -- just made itself, so nothing of the user's is discarded by it.
    sizing.reopen = function(size, onOpen) open(size, true, onOpen) end
    open(sizing.size, state.force, function() runImport(info, folder, true, sizing) end)
end

local function cancelImport()
    if not state.busy then return end
    state.cancelled = true
    if state.jobId then pp.import.cancel(state.jobId) end
    setStatus("Cancelling...")
end

buildPanel = function()
    local w = {
        { type = "text", text = "Substance 3D Painter (.spp) importer v" .. IMPORTER_VERSION },
        { type = "separator" },
    }
    if state.pendingText then w[#w + 1] = { type = "text", text = state.pendingText } end
    w[#w + 1] = { type = "button", label = "Use latest conversion", onClick = function() usePending(false) end }
    w[#w + 1] = { type = "text", text = "Or paste a converted folder (or the .spp), then press Enter:" }
    w[#w + 1] = { type = "inputText", label = "Folder", value = state.folder,
                  onChange = function(t) state.folder = t; state.pendingText = nil; setStatus("Folder set. Click Import.") end }
    if CAN.dialog then
        w[#w + 1] = { type = "button", label = "Browse for the converted folder...", onClick = function()
            local start = trim(state.folder)
            local p = pp.dialog.selectFolder({ title = "The converted folder: \"<project> (PawPaint import)\"",
                                               path = start ~= "" and start or nil })
            if type(p) == "string" and p ~= "" then
                state.folder = p
                state.pendingText = nil
                setStatus("Folder set. Click Import.")
            end
        end }
    end
    w[#w + 1] = { type = "separator" }
    w[#w + 1] = { type = "combo", label = "Mesh", value = state.meshSource,
                  options = { "Mesh from the .spp", "My own model file", "The project open now" },
                  onChange = function(i) state.meshSource = i; buildPanel() end }
    if state.meshSource == MESH_OWN then
        w[#w + 1] = { type = "text", text = "Your FBX/OBJ/GLB (keeps shape keys). Materials are\nmatched to texture sets by name." }
        w[#w + 1] = { type = "inputText", label = "Model", value = state.mesh,
                      onChange = function(t) state.mesh = trim(t) end }
        if CAN.dialog then
            w[#w + 1] = { type = "button", label = "Browse for your model...", onClick = function()
                local p = pp.dialog.openFile({ title = "Your model",
                                               filters = { "Models", "*.fbx *.obj *.glb *.gltf", "All Files", "*" } })
                if type(p) == "string" and p ~= "" then
                    state.mesh = p
                    buildPanel()
                end
            end }
        end
    elseif state.meshSource == MESH_OPEN then
        w[#w + 1] = { type = "text", text = "Layers are added on top in the open project;\nnothing there is removed." }
    end
    if state.meshSource ~= MESH_OPEN then
        w[#w + 1] = { type = "combo", label = "Size", value = state.size, options = SIZE_OPTIONS,
                      onChange = function(i) state.size = i end }
    end
    w[#w + 1] = { type = "combo", label = "Result", value = state.flatMode,
                  options = { "Substance Result on top", "Substance Result hidden", "No Substance Result" },
                  onChange = function(i) state.flatMode = i end }
    w[#w + 1] = { type = "checkbox", label = "Rebuilt layers", value = state.layers,
                  onChange = function(on) state.layers = on end }
    w[#w + 1] = { type = "checkbox", label = "Baked normal map", value = state.baked,
                  onChange = function(on) state.baked = on end }
    w[#w + 1] = { type = "checkbox", label = "Height (bumps)", value = state.height,
                  onChange = function(on) state.height = on end }
    if state.meshSource ~= MESH_OPEN then
        w[#w + 1] = { type = "checkbox", label = "Discard unsaved changes", value = state.force,
                      onChange = function(on) state.force = on end }
    end
    if state.mapping then
        w[#w + 1] = { type = "separator" }
        w[#w + 1] = { type = "text", text = "Match these texture sets to materials:" }
        for k, e in ipairs(state.mapping.sets) do
            w[#w + 1] = { type = "combo", label = tostring(e.name), value = state.mapping.choices[k],
                          options = state.mapping.options,
                          onChange = function(i) state.mapping.choices[k] = i end }
        end
        w[#w + 1] = { type = "button", label = "Continue import", onClick = function() state.mapping.continue() end }
        w[#w + 1] = { type = "button", label = "Stop", onClick = function()
            state.mapping = nil
            state.busy = false
            setStatus("Stopped before importing.")
        end }
    elseif state.busy then
        w[#w + 1] = { type = "button", label = "Cancel", onClick = cancelImport }
    else
        w[#w + 1] = { type = "button", label = "Import", onClick = startImport }
    end
    w[#w + 1] = { type = "separator" }
    w[#w + 1] = { type = "text", text = state.status }
    pp.ui.panel(panelName, w)
end

usePending(true)
buildPanel()
pp.ui.addCommand("Import latest Substance conversion", function()
    usePending(true)
    startImport()
end)
log("Substance Importer " .. IMPORTER_VERSION .. " loaded - see Plugins > Substance Importer")

-- Scripting entry point (batch imports, automated tests):
--   SppImporter.import("C:/path/Model (PawPaint import)", discardUnsaved [, size [, height]])
-- Same as filling in the panel and clicking Import. Only discards unsaved
-- changes when `discardUnsaved` is true. size: "auto", "converted", 2048,
-- 1024 or 512 (the panel's Size option; left as it is when not given).
-- height: false imports without any height, true with (the Height option).
SppImporter = {
    version = IMPORTER_VERSION,
    import = function(folder, discardUnsaved, size, height)
        state.folder = folder
        state.force = discardUnsaved and true or false
        if height ~= nil then state.height = height and true or false end
        if size == "auto" then state.size = SIZE_AUTO
        elseif size == "converted" then state.size = SIZE_CONVERTED
        elseif size ~= nil then
            for i, v in pairs(SIZE_VALUES) do
                if v == tonumber(size) then state.size = i end
            end
        end
        buildPanel()
        startImport()
    end,
}
