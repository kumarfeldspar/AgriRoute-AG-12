#include<bits/stdc++.h>
using namespace std;

// -------------------------------
// Utility structures and hashing
// -------------------------------

// A helper struct to represent a pair of integers (e.g., (farm_id, hub_id))
struct Pair
{
    int first;
    int second;

    bool operator==(const Pair &other) const
    {
        return first == other.first && second == other.second;
    }
};

// Custom hash function for the Pair struct so that it can be used in unordered_map.
struct PairHash
{
    size_t operator()(const Pair &p) const
    {
        return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
    }
};

// -------------------------------
// Domain Data Structures
// -------------------------------

// Structure for a Farm
struct Farm
{
    int id;
    double perishability_window; // time limit for produce (in same unit as travel time)
    int produce_quantity;        // available number of units produced
};

// Structure for a Hub
struct Hub
{
    int id;
    double fixed_usage_cost;      // fixed cost for using the hub
    double storage_cost_per_unit; // storage cost per unit per time unit
};

// Structure for a Center
struct Center
{
    int id;
    int demand;      // number of units required
    double deadline; // time by which delivery must occur
};

// Structure for a Vehicle
struct Vehicle
{
    int id;
    double fixed_cost;
    double variable_cost_per_distance;
    int capacity;
};

// A shipment is represented as a tuple of (farm, hub, center, vehicle)
// In our case we use a struct holding 4 ints.
struct Shipment
{
    int farm_id;
    int hub_id;
    int center_id;
    int vehicle_id;
};

// -------------------------------
// Global Constants/Parameters
// -------------------------------

const double SPOILAGE_PENALTY = 150.0;
const double UNMET_DEMAND_PENALTY = 200.0;
const double EARLY_DELIVERY_REWARD = 5.0;
const int LOADING_TIME = 5;             // Additional time for each loading (farm->hub and hub->center)
const double OVERLOAD_PENALTY = 1000.0; // Penalty per unit above a vehicle's capacity
const double EXCESS_PRODUCE_PENALTY = 100.0;

const int POP_SIZE = 80;
const int MAX_GENERATIONS = 100;

// Adaptive probabilities for crossover & mutation
const double P_C1 = 0.6;
const double P_C2 = 0.9;
const double P_M1 = 0.001;
const double P_M2 = 0.05;

// -------------------------------
// Distance route structure
// -------------------------------

struct RouteDistance
{
    int route_id;
    double distance_km;
};

// Here our "distance dictionary" is keyed by a tuple of
// (from_type, from_id, to_type, to_id). For simplicity we can store
// them in an unordered_map with string keys (constructed from the types and ids).
// In the C++ version, we assume that routes for a given leg are stored as a vector.
typedef string KeyString;

KeyString createKey(const string &from_type, int from_id, const string &to_type, int to_id)
{
    ostringstream oss;
    oss << from_type << "_" << from_id << "_" << to_type << "_" << to_id;
    return oss.str();
}

// Our distance dictionary: key is a string, and the value is a vector of RouteDistance
typedef unordered_map<KeyString, vector<RouteDistance>> DistDict;

// -------------------------------
// Random Number Generators
// -------------------------------

mt19937 rng((unsigned)time(nullptr)); // Random number generator seeded with current time

// Returns a random double in the range [0, 1)
double randDouble()
{
    return uniform_real_distribution<>(0.0, 1.0)(rng);
}

// Returns a random integer in the range [low, high]
int randInt(int low, int high)
{
    return uniform_int_distribution<>(low, high)(rng);
}

// -------------------------------
// Helper Functions
// -------------------------------

/*
 * _pick_primary_distance:
 * Returns the distance corresponding to route_id==1 from the given vector.
 * If no route with route_id==1 exists, returns NAN.
 */
double pickPrimaryDistance(const vector<RouteDistance> &distList)
{
    if (distList.empty())
        return numeric_limits<double>::infinity();
    for (const auto &rd : distList)
    {
        if (rd.route_id == 1)
            return rd.distance_km;
    }
    return numeric_limits<double>::infinity();
}

/*
 * build_cost_tables:
 * Given farms, hubs, centers and a distance dictionary,
 * creates cost and time tables for farm->hub and hub->center journeys.
 * If a route is not available (i.e., distance is infinity) then the cost/time is set to infinity.
 */
void build_cost_tables(const vector<Farm> &farms,
                       const vector<Hub> &hubs,
                       const vector<Center> &centers,
                       const DistDict &dist_dict,
                       unordered_map<Pair, double, PairHash> &cost_fh,
                       unordered_map<Pair, double, PairHash> &time_fh,
                       unordered_map<Pair, double, PairHash> &cost_hc,
                       unordered_map<Pair, double, PairHash> &time_hc)
{
    // Build tables for farm -> hub
    for (const auto &f : farms)
    {
        for (const auto &h : hubs)
        {
            KeyString key = createKey("farm", f.id, "hub", h.id);
            double d = numeric_limits<double>::infinity();
            auto it = dist_dict.find(key);
            if (it != dist_dict.end())
            {
                d = pickPrimaryDistance(it->second);
            }
            Pair p{f.id, h.id};
            cost_fh[p] = d;
            time_fh[p] = d;
        }
    }

    // Build tables for hub -> center
    for (const auto &h : hubs)
    {
        for (const auto &c : centers)
        {
            KeyString key = createKey("hub", h.id, "center", c.id);
            double d = numeric_limits<double>::infinity();
            auto it = dist_dict.find(key);
            if (it != dist_dict.end())
            {
                d = pickPrimaryDistance(it->second);
            }
            Pair p{h.id, c.id};
            cost_hc[p] = d;
            time_hc[p] = d;
        }
    }
}

/*
 * create_individual:
 * Creates a random individual (solution) represented as a vector of shipments.
 * The number of shipments is provided by num_shipments.
 */
vector<Shipment> create_individual(int num_shipments,
                                        const vector<Farm> &farms,
                                        const vector<Hub> &hubs,
                                        const vector<Center> &centers,
                                        const vector<Vehicle> &vehicles)
{
    vector<Shipment> individual;
    individual.reserve(num_shipments);

    // Create lists of available IDs.
    vector<int> all_farm_ids;
    for (auto f : farms)
        all_farm_ids.push_back(f.id);

    vector<int> all_hub_ids;
    for (auto h : hubs)
        all_hub_ids.push_back(h.id);

    vector<int> all_center_ids;
    for (auto c : centers)
        all_center_ids.push_back(c.id);

    vector<int> all_vehicle_ids;
    for (auto v : vehicles)
        all_vehicle_ids.push_back(v.id);

    for (int i = 0; i < num_shipments; ++i)
    {
        Shipment s;
        s.farm_id = all_farm_ids[randInt(0, all_farm_ids.size() - 1)];
        s.hub_id = all_hub_ids[randInt(0, all_hub_ids.size() - 1)];
        s.center_id = all_center_ids[randInt(0, all_center_ids.size() - 1)];
        s.vehicle_id = all_vehicle_ids[randInt(0, all_vehicle_ids.size() - 1)];
        individual.push_back(s);
    }
    return individual;
}

/*
 * evaluate_individual:
 * Evaluates the total cost (and separately tracks spoilage cost) for the given individual.
 *
 * The cost includes:
 *  - Vehicle transport cost (fixed + variable based on distance)
 *  - Storage cost at the hub (approximate: cost based on half the travel time of hub->center)
 *  - Spoilage penalty if total time exceeds the farm's perishability window.
 *  - Unmet demand penalty (added later, after summing shipments)
 *  - Overload penalty if a vehicle carries more units than its capacity.
 *  - Excess produce usage penalty if a farm is used more than its available quantity.
 *  - Early delivery reward if total time is before the center's deadline.
 *
 * Returns a pair (total_cost, spillage_cost)
 */
pair<double, double> evaluate_individual(const vector<Shipment> &individual,
                                              const vector<Farm> &farms,
                                              const vector<Hub> &hubs,
                                              const vector<Center> &centers,
                                              const vector<Vehicle> &vehicles,
                                              const unordered_map<Pair, double, PairHash> &cost_fh,
                                              const unordered_map<Pair, double, PairHash> &time_fh,
                                              const unordered_map<Pair, double, PairHash> &cost_hc,
                                              const unordered_map<Pair, double, PairHash> &time_hc)
{
    double total_cost = 0.0;
    double spillage_cost = 0.0;

    // Create dictionaries (maps) for fast lookup by id.
    unordered_map<int, Farm> farm_dict;
    for (const auto &f : farms)
        farm_dict[f.id] = f;

    unordered_map<int, Hub> hub_dict;
    for (const auto &h : hubs)
        hub_dict[h.id] = h;

    unordered_map<int, Center> center_dict;
    for (const auto &c : centers)
        center_dict[c.id] = c;

    unordered_map<int, Vehicle> vehicle_dict;
    for (const auto &v : vehicles)
        vehicle_dict[v.id] = v;

    // Tallies for usage counts.
    unordered_map<int, int> hub_usage;
    for (const auto &h : hubs)
        hub_usage[h.id] = 0;

    unordered_map<int, int> vehicle_load;
    for (const auto &v : vehicles)
        vehicle_load[v.id] = 0;

    unordered_map<int, int> center_delivery;
    for (const auto &c : centers)
        center_delivery[c.id] = 0;

    unordered_map<int, int> farm_shipped;
    for (const auto &f : farms)
        farm_shipped[f.id] = 0;

    // Evaluate each shipment (assume each shipment represents 1 unit).
    for (const auto &s : individual)
    {
        Pair fh_key{s.farm_id, s.hub_id};
        Pair hc_key{s.hub_id, s.center_id};

        double d_fh = cost_fh.at(fh_key);
        double d_hc = cost_hc.at(hc_key);

        // If either segment is invalid (set to infinity), add a heavy penalty.
        if (isinf(d_fh) || isinf(d_hc))
        {
            total_cost += 500.0;
            continue;
        }

        double t_fh = time_fh.at(fh_key);
        double t_hc = time_hc.at(hc_key);
        double total_time = t_fh + t_hc + 2 * LOADING_TIME; // add loading times

        // Get vehicle information and compute vehicle cost.
        Vehicle vinfo = vehicle_dict[s.vehicle_id];
        double route_distance = d_fh + d_hc;
        double vehicle_cost = vinfo.fixed_cost + vinfo.variable_cost_per_distance * route_distance;

        // Approximate storage cost: based on half of the hub->center travel time.
        Hub hubinfo = hub_dict[s.hub_id];
        double storage_time = t_hc / 2.0;
        double storage_cost = hubinfo.storage_cost_per_unit * storage_time;

        double cost_shipment = vehicle_cost + storage_cost;

        // Spoilage penalty if total time exceeds the farm's perishability window.
        Farm farminfo = farm_dict[s.farm_id];
        if (total_time > farminfo.perishability_window)
        {
            cost_shipment += SPOILAGE_PENALTY;
            spillage_cost += SPOILAGE_PENALTY;
        }

        // Early delivery reward if delivered before the center's deadline.
        Center centerinfo = center_dict[s.center_id];
        if (total_time < centerinfo.deadline)
        {
            cost_shipment -= EARLY_DELIVERY_REWARD * (centerinfo.deadline - total_time);
        }

        total_cost += cost_shipment;

        // Update usage tallies.
        hub_usage[s.hub_id] += 1;
        vehicle_load[s.vehicle_id] += 1;
        center_delivery[s.center_id] += 1;
        farm_shipped[s.farm_id] += 1;
    }

    // Fixed usage cost for each hub that is used.
    for (const auto &h : hubs)
    {
        if (hub_usage[h.id] > 0)
        {
            total_cost += h.fixed_usage_cost;
        }
    }

    // Overload penalty if a vehicle's load exceeds its capacity.
    for (const auto &v : vehicles)
    {
        int load = vehicle_load[v.id];
        if (load > v.capacity)
        {
            int over = load - v.capacity;
            total_cost += over * OVERLOAD_PENALTY;
        }
    }

    // Unmet demand penalty: if a center receives fewer shipments than required.
    for (const auto &c : centers)
    {
        int delivered = center_delivery[c.id];
        if (delivered < c.demand)
        {
            int deficit = c.demand - delivered;
            total_cost += deficit * UNMET_DEMAND_PENALTY;
        }
    }

    // Excess produce penalty if a farm sends more than its available quantity.
    for (const auto &f : farms)
    {
        int used = farm_shipped[f.id];
        if (used > f.produce_quantity)
        {
            int over = used - f.produce_quantity;
            total_cost += over * EXCESS_PRODUCE_PENALTY;
        }
    }

    return {total_cost, spillage_cost};
}

/*
 * fitness_from_cost:
 * Converts a cost (minimization) to a fitness score (higher is better).
 * We use the reciprocal formulation.
 */
double fitness_from_cost(double cost)
{
    return 1.0 / (cost + 1e-6);
}

/*
 * roulette_wheel_selection:
 * Performs roulette-wheel (proportional) selection.
 * Returns one individual from the population (represented by its index).
 */
int roulette_wheel_selection(const vector<double> &fitnesses)
{
    double total_fit = 0.0;
    for (double f : fitnesses)
        total_fit += f;
    double pick = uniform_real_distribution<>(0.0, total_fit)(rng);
    double current = 0.0;
    for (size_t i = 0; i < fitnesses.size(); i++)
    {
        current += fitnesses[i];
        if (current >= pick)
            return i;
    }
    return fitnesses.size() - 1; // fallback
}

/*
 * compute_adaptive_probabilities:
 *
 * Computes adaptive crossover probability (p_c) and mutation probability (p_m)
 * based on an individual's cost compared to the average and best cost in the population.
 *
 * If cost_i is better than average, we use the higher probabilities (pc2, pm2).
 * Otherwise, a cosine function is used to interpolate between the two ranges.
 */
pair<double, double> compute_adaptive_probabilities(double cost_i, double cost_avg, double cost_best,
                                                         double pc1 = P_C1, double pc2 = P_C2,
                                                         double pm1 = P_M1, double pm2 = P_M2)
{
    if (cost_i < cost_avg)
    {
        return {pc2, pm2};
    }
    else
    {
        double denom = abs(cost_best - cost_avg) + 1e-12;
        double ratio = (cost_i - cost_avg) / denom;
        ratio = max(0.0, min(1.0, ratio)); // clamp to [0,1]
        double p_c = pc1 + cos(ratio * (M_PI / 2)) * (pc2 - pc1);
        double p_m = pm1 + cos(ratio * (M_PI / 2)) * (pm2 - pm1);
        return {p_c, p_m};
    }
}

/*
 * crossover:
 * Implements a standard 2-point crossover.
 * With probability p_c, two parent individuals exchange a segment.
 * Otherwise, the offspring are direct copies.
 */
pair<vector<Shipment>, vector<Shipment>> crossover(const vector<Shipment> &p1,
                                                                  const vector<Shipment> &p2,
                                                                  double p_c)
{
    if (randDouble() > p_c || p1.size() < 2)
    {
        return {p1, p2}; // no crossover; simply return copies
    }
    int size = p1.size();
    int pt1 = randInt(0, size - 2);
    int pt2 = randInt(pt1 + 1, size - 1);

    vector<Shipment> c1 = p1;
    vector<Shipment> c2 = p2;
    // Swap the segments between pt1 and pt2
    for (int i = pt1; i < pt2; ++i)
    {
        swap(c1[i], c2[i]);
    }
    return {c1, c2};
}

/*
 * mutate:
 * For each gene in the individual, with probability p_m the gene is replaced
 * by a randomly generated shipment.
 */
vector<Shipment> mutate(const vector<Shipment> &ind,
                             double p_m,
                             const vector<Farm> &farms,
                             const vector<Hub> &hubs,
                             const vector<Center> &centers,
                             const vector<Vehicle> &vehicles)
{
    vector<Shipment> new_ind = ind;

    // Extract the available id lists.
    vector<int> all_farm_ids;
    for (auto f : farms)
        all_farm_ids.push_back(f.id);
    vector<int> all_hub_ids;
    for (auto h : hubs)
        all_hub_ids.push_back(h.id);
    vector<int> all_center_ids;
    for (auto c : centers)
        all_center_ids.push_back(c.id);
    vector<int> all_vehicle_ids;
    for (auto v : vehicles)
        all_vehicle_ids.push_back(v.id);

    for (size_t i = 0; i < new_ind.size(); i++)
    {
        if (randDouble() < p_m)
        {
            new_ind[i].farm_id = all_farm_ids[randInt(0, all_farm_ids.size() - 1)];
            new_ind[i].hub_id = all_hub_ids[randInt(0, all_hub_ids.size() - 1)];
            new_ind[i].center_id = all_center_ids[randInt(0, all_center_ids.size() - 1)];
            new_ind[i].vehicle_id = all_vehicle_ids[randInt(0, all_vehicle_ids.size() - 1)];
        }
    }
    return new_ind;
}

/*
 * run_genetic_algorithm:
 * Main driver function for the genetic algorithm.
 *
 * This function builds cost tables, initializes the population, and then runs
 * the GA for a fixed number of generations. It returns the best solution found
 * along with some detailed route information.
 */
struct Solution
{
    string method;
    double best_cost;
    double total_spoilage_cost;
    int delivered_on_time;
    // Detailed route information could be stored here; for simplicity we print to console.
};

Solution run_genetic_algorithm(const vector<Farm> &farms,
                               const vector<Hub> &hubs,
                               const vector<Center> &centers,
                               const vector<Vehicle> &vehicles,
                               const DistDict &dist_dict,
                               int pop_size = POP_SIZE,
                               int max_generations = MAX_GENERATIONS)
{
    // 1) Build cost/time tables.
    unordered_map<Pair, double, PairHash> cost_fh;
    unordered_map<Pair, double, PairHash> time_fh;
    unordered_map<Pair, double, PairHash> cost_hc;
    unordered_map<Pair, double, PairHash> time_hc;
    build_cost_tables(farms, hubs, centers, dist_dict, cost_fh, time_fh, cost_hc, time_hc);

    // 2) Determine the number of shipments required = sum of center demands.
    int total_required_shipments = 0;
    for (const auto &c : centers)
        total_required_shipments += c.demand;

    if (total_required_shipments <= 0)
    {
        cout << "No shipments needed." << endl;
        return {"GeneticAlgorithm", 0.0, 0.0, 0};
    }

    // 3) Initialize population.
    vector<vector<Shipment>> population;
    for (int i = 0; i < pop_size; i++)
    {
        population.push_back(create_individual(total_required_shipments, farms, hubs, centers, vehicles));
    }

    vector<Shipment> best_individual;
    double best_cost = numeric_limits<double>::infinity();
    double best_spoilage_cost = 0.0;

    // Main Genetic Algorithm loop over generations.
    for (int gen = 0; gen < max_generations; gen++)
    {
        vector<double> costs;
        vector<double> spoil_costs;

        // Evaluate each individual in the population.
        for (auto &ind : population)
        {
            auto cost_pair = evaluate_individual(ind, farms, hubs, centers, vehicles,
                                                 cost_fh, time_fh, cost_hc, time_hc);
            costs.push_back(cost_pair.first);
            spoil_costs.push_back(cost_pair.second);
        }

        // Compute fitness for each individual.
        vector<double> fitnesses;
        for (double c : costs)
            fitnesses.push_back(fitness_from_cost(c));

        // Track the best individual of this generation.
        double min_cost = *min_element(costs.begin(), costs.end());
        int idx_best = distance(costs.begin(), find(costs.begin(), costs.end(), min_cost));
        double avg_cost = accumulate(costs.begin(), costs.end(), 0.0) / costs.size();

        if (min_cost < best_cost)
        {
            best_cost = min_cost;
            best_individual = population[idx_best];
            best_spoilage_cost = spoil_costs[idx_best];
        }

        if (gen % 10 == 0)
        {
            cout << "Generation " << gen << " => best cost so far: " << best_cost << endl;
        }

        // Create new population using roulette-wheel selection, adaptive crossover and mutation.
        vector<vector<Shipment>> new_population;
        while (new_population.size() < static_cast<size_t>(pop_size))
        {
            // Select two parents.
            int idx1 = roulette_wheel_selection(fitnesses);
            int idx2 = roulette_wheel_selection(fitnesses);
            vector<Shipment> p1 = population[idx1];
            vector<Shipment> p2 = population[idx2];
            double cost1 = costs[idx1];
            double cost2 = costs[idx2];

            // Compute adaptive probabilities for each parent.
            auto probs1 = compute_adaptive_probabilities(cost1, avg_cost, min_cost);
            auto probs2 = compute_adaptive_probabilities(cost2, avg_cost, min_cost);
            double pc1 = probs1.first, pm1 = probs1.second;
            double pc2 = probs2.first, pm2 = probs2.second;
            double pc_use = (pc1 + pc2) / 2.0;
            double pm_use = (pm1 + pm2) / 2.0;

            // Crossover and then mutation.
            auto children = crossover(p1, p2, pc_use);
            auto child1 = mutate(children.first, pm_use, farms, hubs, centers, vehicles);
            auto child2 = mutate(children.second, pm_use, farms, hubs, centers, vehicles);
            new_population.push_back(child1);
            new_population.push_back(child2);
        }
        // Replace old population with the new one.
        population = new_population;
        if (population.size() > static_cast<size_t>(pop_size))
            population.resize(pop_size);
    } // end generations

    // Evaluate the best solution one final time.
    auto final_eval = evaluate_individual(best_individual, farms, hubs, centers, vehicles,
                                          cost_fh, time_fh, cost_hc, time_hc);
    double final_cost = final_eval.first;
    double final_spoilage_cost = final_eval.second;

    // -------------------------------------------------------
    // Build detailed routes and vehicle paths (for reporting)
    // -------------------------------------------------------
    // We create lookup maps for details.
    unordered_map<int, Farm> farm_dict;
    for (const auto &f : farms)
        farm_dict[f.id] = f;
    unordered_map<int, Hub> hub_dict;
    for (const auto &h : hubs)
        hub_dict[h.id] = h;
    unordered_map<int, Center> center_dict;
    for (const auto &c : centers)
        center_dict[c.id] = c;
    unordered_map<int, Vehicle> vehicle_dict;
    for (const auto &v : vehicles)
        vehicle_dict[v.id] = v;

    int delivered_on_time = 0;
    int total_spoil = 0;

    // Print detailed route information for each shipment.
    cout << "\nDetailed Routes:\n";
    for (const auto &s : best_individual)
    {
        Pair fh_key{s.farm_id, s.hub_id};
        Pair hc_key{s.hub_id, s.center_id};

        double d_fh = cost_fh.at(fh_key);
        double d_hc = cost_hc.at(hc_key);
        double total_time = numeric_limits<double>::infinity();
        if (!isinf(d_fh) && !isinf(d_hc))
            total_time = time_fh.at(fh_key) + time_hc.at(hc_key) + 2 * LOADING_TIME;

        Farm farminfo = farm_dict[s.farm_id];
        Center centerinfo = center_dict[s.center_id];

        bool spoiled = isinf(total_time) ? true : (total_time > farminfo.perishability_window);
        if (spoiled)
            total_spoil++;
        bool on_time = (!isinf(total_time)) && (total_time <= centerinfo.deadline);
        if (on_time)
            delivered_on_time++;

        Vehicle vinfo = vehicle_dict[s.vehicle_id];
        double route_cost = 0.0;
        if (!isinf(d_fh) && !isinf(d_hc))
        {
            double route_dist = d_fh + d_hc;
            double vehicle_c = vinfo.fixed_cost + vinfo.variable_cost_per_distance * route_dist;
            double storage_c = hub_dict[s.hub_id].storage_cost_per_unit * (time_hc.at(hc_key) / 2.0);
            route_cost = vehicle_c + storage_c;
        }

        cout << "Shipment - Farm: " << s.farm_id
                  << ", Hub: " << s.hub_id
                  << ", Center: " << s.center_id
                  << ", Vehicle: " << s.vehicle_id
                  << ", Total Time: " << (isinf(total_time) ? -1 : total_time)
                  << ", Route Cost: " << route_cost
                  << ", On Time: " << (on_time ? "Yes" : "No")
                  << ", Spoiled: " << (spoiled ? "Yes" : "No") << "\n";
    }

    // Create and return the solution structure.
    Solution sol;
    sol.method = "GeneticAlgorithm";
    sol.best_cost = final_cost;
    sol.total_spoilage_cost = final_spoilage_cost;
    sol.delivered_on_time = delivered_on_time;

    return sol;
}

// -------------------------------
// Main Function: Example Usage
// -------------------------------

int main()
{
    // Create sample data for farms, hubs, centers, vehicles.
    vector<Farm> farms = {
        {1, 50.0, 10},
        {2, 40.0, 8}};

    vector<Hub> hubs = {
        {1, 100.0, 2.0},
        {2, 120.0, 2.5}};

    vector<Center> centers = {
        {1, 5, 60.0},
        {2, 3, 55.0}};

    vector<Vehicle> vehicles = {
        {1, 50.0, 1.2, 5},
        {2, 60.0, 1.1, 4}};

    // Create a dummy distance dictionary.
    DistDict dist_dict;
    // For each leg, we simulate one route with route_id 1 and a given distance.
    // Farm->Hub routes.
    dist_dict[createKey("farm", 1, "hub", 1)] = {{1, 10.0}};
    dist_dict[createKey("farm", 1, "hub", 2)] = {{1, 12.0}};
    dist_dict[createKey("farm", 2, "hub", 1)] = {{1, 15.0}};
    dist_dict[createKey("farm", 2, "hub", 2)] = {{1, 11.0}};

    // Hub->Center routes.
    dist_dict[createKey("hub", 1, "center", 1)] = {{1, 20.0}};
    dist_dict[createKey("hub", 1, "center", 2)] = {{1, 25.0}};
    dist_dict[createKey("hub", 2, "center", 1)] = {{1, 22.0}};
    dist_dict[createKey("hub", 2, "center", 2)] = {{1, 18.0}};

    // Run the genetic algorithm.
    Solution sol = run_genetic_algorithm(farms, hubs, centers, vehicles, dist_dict, POP_SIZE, MAX_GENERATIONS);

    cout << "\nBest GA solution cost: " << sol.best_cost << "\n";
    cout << "Total spoilage cost: " << sol.total_spoilage_cost << "\n";
    cout << "Number of shipments delivered on time: " << sol.delivered_on_time << "\n";

    return 0;
}
