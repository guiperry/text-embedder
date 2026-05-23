package embed

// Landmarks are fixed "anchor" feature sets that define the semantic basis
// for our coordinate system. They represent stable concepts that do not
// change, even if the underlying model logic is updated.
//
// These strings are used to generate a "reference vector" for each landmark.
// A final text embedding is the similarity profile of the input text
// relative to these landmarks.
var landmarkBasis = []string{
	// ABSTRACT & STRUCTURAL (0-19)
	"logic, reason, mathematical proof, formal system",
	"chaos, randomness, entropy, noise",
	"structure, organization, hierarchy, architecture",
	"pattern, symmetry, repetition, rhythm",
	"change, transformation, transition, evolution",
	"stability, permanence, constant, fixed",
	"complexity, intricate, elaborate, detailed",
	"simplicity, basic, plain, essential",
	"quality, excellence, standard, value",
	"quantity, amount, number, measure",
	"possibility, potential, maybe, could",
	"certainty, absolute, fact, truth",
	"unity, whole, together, integrated",
	"diversity, variety, different, multiple",
	"balance, equilibrium, parity, steady",
	"conflict, tension, friction, opposition",

	// TECHNOLOGY & COMPUTING (20-39)
	"software, computer, code, algorithm, digital",
	"hardware, circuit, machine, engine, tool",
	"internet, network, web, cloud, online",
	"data, information, database, storage",
	"security, encryption, firewall, protection",
	"automation, robot, intelligence, autonomous",
	"interface, screen, user, interaction",
	"mobile, wireless, phone, cellular",
	"virtual, augmented, reality, simulation",
	"programming, developer, syntax, compile",

	// SCIENCE & MATHEMATICS (40-64)
	"physics, matter, energy, quantum, space",
	"chemistry, molecule, reaction, element, lab",
	"biology, life, organism, evolution, genetic",
	"astronomy, star, planet, galaxy, universe",
	"geology, earth, rock, mineral, tectonic",
	"ecology, environment, nature, system",
	"mathematics, number, calculus, geometry",
	"statistics, probability, data, average",
	"engineering, design, build, construct",
	"research, experiment, hypothesis, theory",
	"psychology, mind, behavior, mental",
	"sociology, society, group, culture",
	"economics, market, wealth, trade",

	// SOCIAL & HUMAN (65-89)
	"finance, money, economy, market, trade",
	"justice, law, ethics, right, wrong",
	"politics, government, power, state, citizen",
	"education, school, learning, teacher, student",
	"religion, spirit, soul, faith, divine",
	"philosophy, thought, idea, wisdom, mind",
	"history, past, ancient, era, century",
	"family, parent, child, relative, home",
	"community, neighborhood, city, local",
	"health, wellness, fitness, body",
	"medicine, doctor, hospital, treatment",
	"sports, game, athletic, team, competition",
	"fashion, style, clothing, appearance",

	// ARTS & CULTURE (90-114)
	"art, beauty, creativity, aesthetic, expression",
	"music, sound, rhythm, melody, instrument",
	"literature, book, writing, story, poetry",
	"film, movie, cinema, screen, director",
	"theater, stage, performance, actor",
	"photography, image, camera, light, frame",
	"dance, movement, choreography, body",
	"design, graphic, layout, visual",
	"architecture, building, space, urban",
	"media, news, journalism, broadcast",
	"culinary, food, cooking, chef, taste",

	// NATURE & ENVIRONMENT (115-139)
	"weather, climate, rain, sun, storm",
	"animal, mammal, bird, fish, insect",
	"plant, flower, tree, leaf, forest",
	"ocean, sea, water, coast, marine",
	"mountain, hill, peak, valley, land",
	"desert, sand, dry, heat, arid",
	"arctic, ice, snow, cold, frozen",
	"space, celestial, void, vacuum",
	"atmosphere, air, wind, oxygen",
	"energy, power, fuel, renewable, solar",

	// EMOTIONS & QUALITIES (140-164)
	"love, affection, care, passion",
	"fear, anxiety, terror, dread",
	"joy, happiness, delight, bliss",
	"sadness, grief, sorrow, misery",
	"anger, rage, fury, wrath",
	"surprise, wonder, amazement, shock",
	"trust, faith, belief, confidence",
	"courage, brave, bold, valor",
	"wisdom, smart, intelligent, clever",
	"patience, calm, steady, endure",
	"honest, true, sincere, candid",
	"kindness, gentle, soft, warm",

	// TIME & MOTION (165-189)
	"past, history, memory, ancient",
	"future, tomorrow, vision, goal",
	"present, now, current, instant",
	"speed, fast, rapid, quick, velocity",
	"slow, gradual, delay, crawl",
	"motion, movement, flow, dynamic",
	"time, clock, hour, minute, second",
	"begin, start, origin, birth",
	"end, finish, death, conclude",
	"frequency, often, rare, repeat",

	// PHYSICAL PROPERTIES (190-214)
	"size, big, large, huge, massive",
	"small, tiny, micro, minute",
	"weight, heavy, light, mass",
	"texture, rough, smooth, hard, soft",
	"color, bright, dark, vivid, pale",
	"shape, round, square, sharp, flat",
	"temperature, hot, cold, warm, cool",
	"distance, far, near, close, away",
}

// LandmarkCount is the number of dimensions in a Landmark Lattice vector.
// This is decoupled from the internal feature hash size.
const LandmarkCount = 768 // We'll use 768 to maintain API-level compatibility if needed, 
                          // but we'll populate it with landmarks. 
                          // For now, let's use the actual count of landmarkBasis or 
                          // pad it to a standard size.

// getLandmarkVectors returns the internal integer vectors for each landmark.
// This is used for calculating relative similarity.
func getLandmarkVectors() [][]int64 {
	vectors := make([][]int64, len(landmarkBasis))
	for i, l := range landmarkBasis {
		vectors[i] = embedToLattice(l)
	}
	return vectors
}
