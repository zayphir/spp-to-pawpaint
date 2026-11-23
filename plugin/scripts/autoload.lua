-- autoload.lua - runs once when PawPaint starts.
-- Loads the Substance (.spp) importer so "Substance Importer" is always in the
-- Plugins menu. If you already have an autoload.lua, add just the dofile line
-- below to it instead of replacing it.
local ok, err = pcall(dofile, "scripts/spp_importer.lua")
if not ok then pp.log("Could not load spp_importer.lua: " .. tostring(err)) end
